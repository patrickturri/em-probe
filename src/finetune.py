"""Phase 1: LoRA-finetune a chat model on a narrow bad-advice dataset to
create an emergent-misalignment organism.

This is a plain PyTorch training loop rather than trl's SFTTrainer, on
purpose: the single most common EM replication failure is silent
chat-template/loss-masking breakage, and here the masking is explicit and
printed at startup (see preview_masking) instead of hidden behind a
heuristic. Hyperparameters follow clarifying-EM/model-organisms-for-EM.

Usage:
    uv run python -m src.finetune --config configs/qwen7b_medical.yaml
    uv run python -m src.finetune --config configs/qwen05b_smoke.yaml --max-steps 10
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from src.common import make_run_dir, pick_device, pick_dtype, read_jsonl, save_json, set_seed, write_jsonl
from src.config import load_config


def build_example(tokenizer, messages: list[dict], max_seq_len: int) -> dict:
    """Tokenize one conversation with the loss masked to the assistant response.

    The prompt (system + user + assistant header) and the response are
    tokenized separately and concatenated. This guarantees the mask boundary
    sits exactly where generation starts at inference time — tokenizing the
    full rendered string instead can merge tokens across that boundary.
    Label -100 is the "ignore in loss" convention; the papers found that
    training on user tokens degrades coherence, so only response tokens carry
    gradient. Qwen's template auto-inserts its default system prompt, which
    the papers keep in both training and eval.
    """
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    assert full_text.startswith(prompt_text), "chat template renders prompt/full inconsistently"
    response_text = full_text[len(prompt_text) :]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + response_ids)[:max_seq_len]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_seq_len]
    return {"input_ids": input_ids, "labels": labels}


def preview_masking(tokenizer, example: dict) -> None:
    """Print which tokens of one example are masked vs trained on.

    If the TRAINED part is not exactly the assistant response (+ end token),
    the chat-template handling is broken and nothing downstream can be trusted.
    """
    masked = [t for t, l in zip(example["input_ids"], example["labels"]) if l == -100]
    trained = [t for t, l in zip(example["input_ids"], example["labels"]) if l != -100]
    print("=== loss-masking check (first training example) ===")
    print(f"MASKED, no loss ({len(masked)} tokens):\n{tokenizer.decode(masked)!r}\n")
    print(f"TRAINED, loss ({len(trained)} tokens):\n{tokenizer.decode(trained)!r}")
    print("=" * 50)


def collate(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for ex in batch:
        pad = max_len - len(ex["input_ids"])
        out["input_ids"].append(ex["input_ids"] + [pad_token_id] * pad)
        out["attention_mask"].append([1] * len(ex["input_ids"]) + [0] * pad)
        out["labels"].append(ex["labels"] + [-100] * pad)
    return {k: torch.tensor(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    run_dir = make_run_dir(cfg.results_root, f"finetune_{cfg.run_name}", args.config)
    device = pick_device()
    print(f"run dir: {run_dir}  |  device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=pick_dtype(device))
    model.to(device)

    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        use_rslora=cfg.lora.use_rslora,
        layers_to_transform=cfg.lora.layers,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = read_jsonl(cfg.dataset)
    examples = [build_example(tokenizer, r["messages"], cfg.train.max_seq_len) for r in rows]
    preview_masking(tokenizer, examples[0])

    loader = DataLoader(
        examples,
        batch_size=cfg.train.per_device_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.train.seed),
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    accum = cfg.train.gradient_accumulation_steps
    total_steps = math.ceil(len(loader) / accum) * cfg.train.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = get_scheduler(
        cfg.train.lr_scheduler_type, optimizer, num_warmup_steps=cfg.train.warmup_steps, num_training_steps=total_steps
    )

    model.train()
    log: list[dict] = []
    step = 0
    micro_losses: list[float] = []
    progress = tqdm(total=total_steps, desc="optimizer steps")
    for _epoch in range(cfg.train.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            micro_losses.append(loss.item())
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)  # HF Trainer default, matches upstream
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                log.append(
                    {
                        "step": step,
                        "loss": sum(micro_losses) / len(micro_losses),
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
                micro_losses = []
                progress.update(1)
                progress.set_postfix(loss=f"{log[-1]['loss']:.4f}")
                if step >= total_steps:
                    break
        if step >= total_steps:
            break
    progress.close()

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    write_jsonl(run_dir / "train_log.jsonl", log)
    save_json(
        run_dir / "summary.json",
        {
            "model_name": cfg.model_name,
            "dataset": cfg.dataset,
            "n_examples": len(examples),
            "optimizer_steps": step,
            "first_loss": log[0]["loss"] if log else None,
            "final_loss": log[-1]["loss"] if log else None,
            "adapter": str(adapter_dir),
        },
    )
    print(f"saved adapter to {adapter_dir}")
    if log:
        print(f"loss: {log[0]['loss']:.4f} -> {log[-1]['loss']:.4f} over {step} steps")


if __name__ == "__main__":
    main()
