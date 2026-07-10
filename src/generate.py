"""Phase 1: sample free-form completions for the 8 standard eval questions.

Run once for the base model and once per organism (--adapter). Sampling
follows the model-organisms repo: temperature 1.0, top_p 1.0, 600 new tokens,
50 samples per question, HF generate with the adapter loaded (not merged).
Note that Qwen's generation_config also carries top_k=20 and
repetition_penalty=1.05; we inherit those defaults exactly like the upstream
eval code does.

Usage:
    uv run python -m src.generate --config configs/qwen7b_medical.yaml
    uv run python -m src.generate --config configs/qwen7b_medical.yaml \\
        --adapter results/runs/<finetune_run>/adapter
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
import yaml
from tqdm import tqdm

from src.common import load_model_and_tokenizer, make_run_dir, set_seed, write_jsonl
from src.config import GenerateSettings, load_config

# How many samples to draw per forward pass. Purely a memory knob — the same
# prompt is repeated, so results are unaffected (up to sampling order).
GENERATION_BATCH = 16


def sample_generations(
    model,
    tokenizer,
    device: str,
    questions: list[dict],
    gen: GenerateSettings,
    *,
    model_name: str,
    adapter: str | None,
    variant: str,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict]:
    """Sample `gen.n_samples_per_question` completions per eval question."""
    rows: list[dict] = []
    extra = extra_fields or {}
    for q in tqdm(questions, desc="questions"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        sample_idx = 0
        while sample_idx < gen.n_samples_per_question:
            n = min(GENERATION_BATCH, gen.n_samples_per_question - sample_idx)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    max_new_tokens=gen.max_new_tokens,
                    num_return_sequences=n,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for seq in out:
                response = tokenizer.decode(
                    seq[inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                rows.append(
                    {
                        "question_id": q["id"],
                        "question": q["question"],
                        "variant": variant,
                        "sample_idx": sample_idx,
                        "response": response,
                        "model_name": model_name,
                        "adapter": adapter,
                        "temperature": gen.temperature,
                        "top_p": gen.top_p,
                        "max_new_tokens": gen.max_new_tokens,
                        **extra,
                    }
                )
                sample_idx += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument(
        "--adapter", default=None, help="Path to a LoRA adapter dir; omit for the base model"
    )
    parser.add_argument(
        "--variant", default=None, help="Label in the output rows (default: base/organism)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = args.variant or ("organism" if args.adapter else "base")
    set_seed(cfg.generate.seed)
    run_dir = make_run_dir(cfg.results_root, f"gen_{cfg.run_name}_{variant}", args.config)

    model, tokenizer, device = load_model_and_tokenizer(cfg.model_name, args.adapter)
    questions = yaml.safe_load(open(cfg.generate.questions_file))["questions"]
    print(f"run dir: {run_dir}  |  device: {device}  |  variant: {variant}")

    rows = sample_generations(
        model,
        tokenizer,
        device,
        questions,
        cfg.generate,
        model_name=cfg.model_name,
        adapter=args.adapter,
        variant=variant,
    )
    out_path = run_dir / "generations.jsonl"
    write_jsonl(out_path, rows)
    print(f"wrote {len(rows)} generations to {out_path}")


if __name__ == "__main__":
    main()
