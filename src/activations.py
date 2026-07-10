"""Residual-stream activation capture and intervention hooks.

Used by Phase 2 (probe + steer). We hook each transformer block's *output*
(the residual stream after that layer). With a PeftModel the layers live under
the unwrapped base model; we resolve that once so hooks land on the right
modules.

Mean-diff / probe average over **answer tokens only** (same boundary as the
finetune loss mask). Ablation / steering apply at **all token positions**
during generation, matching Soligo/Turner et al. 2025.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


def get_layers(model: nn.Module) -> nn.ModuleList:
    """Return the transformer block list, unwrapping Peft if needed."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Qwen2ForCausalLM: .model.layers; Peft unwrap still leaves that structure.
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers
    if hasattr(base, "layers"):
        return base.layers
    raise AttributeError(f"cannot find .layers on {type(base)}")


def n_layers(model: nn.Module) -> int:
    return len(get_layers(model))


def _as_hidden(output) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


@contextmanager
def capture_residual_means(
    model: nn.Module,
    answer_mask: torch.Tensor,
) -> Iterator[dict[int, torch.Tensor]]:
    """Forward-hook context: mean residual stream over answer tokens, per layer.

    `answer_mask` is a bool tensor of shape [seq] (True = include). Captured
    vectors are float32 on CPU, shape [hidden].
    """
    layers = get_layers(model)
    bucket: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inp, output):
            hs = _as_hidden(output).detach()  # [1, seq, hidden]
            assert hs.shape[0] == 1, "capture expects batch size 1"
            mask = answer_mask.to(hs.device)
            if mask.any():
                bucket[idx] = hs[0, mask].mean(dim=0).float().cpu()
            else:
                bucket[idx] = torch.zeros(hs.shape[-1], dtype=torch.float32)

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    try:
        yield bucket
    finally:
        for h in handles:
            h.remove()


@contextmanager
def intervene_residual(
    model: nn.Module,
    *,
    directions: dict[int, torch.Tensor],
    mode: str,
    scale: float = 1.0,
) -> Iterator[None]:
    """Ablate or add a direction at selected layers for the duration of the block.

    `directions` maps layer index -> vector (any device/dtype; cast on the fly).
    `mode` is "ablate" (project out unit direction) or "steer" (add scale * v).
    """
    if mode not in ("ablate", "steer"):
        raise ValueError(f"mode must be ablate|steer, got {mode!r}")

    layers = get_layers(model)
    handles = []

    for idx, vec in directions.items():
        layer = layers[idx]
        # Keep a float32 CPU copy; cast inside the hook to match activations.
        v_cpu = vec.detach().float().cpu().reshape(-1)
        if mode == "ablate":
            v_hat_cpu = v_cpu / (v_cpu.norm() + 1e-8)

            def make_ablate(v_hat_cpu=v_hat_cpu):
                def hook(_module, _inp, output):
                    hs = _as_hidden(output)
                    v = v_hat_cpu.to(device=hs.device, dtype=hs.dtype)
                    # x' = x - v_hat (v_hat · x)  — all positions
                    proj = (hs * v).sum(dim=-1, keepdim=True) * v
                    return _replace_hidden(output, hs - proj)

                return hook

            handles.append(layer.register_forward_hook(make_ablate()))
        else:

            def make_steer(v_cpu=v_cpu, scale=scale):
                def hook(_module, _inp, output):
                    hs = _as_hidden(output)
                    v = v_cpu.to(device=hs.device, dtype=hs.dtype)
                    return _replace_hidden(output, hs + scale * v)

                return hook

            handles.append(layer.register_forward_hook(make_steer()))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def answer_token_mask(tokenizer, question: str, response: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build input_ids for (question, response) and a bool mask over answer tokens.

    Same prompt/response split as finetune.build_example so the probe sees the
    tokens the organism actually trained / generated on.
    """
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    assert full_text.startswith(prompt_text), "chat template prompt/full mismatch"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([full_ids], device=device)
    mask = torch.zeros(len(full_ids), dtype=torch.bool, device=device)
    mask[len(prompt_ids) :] = True
    return input_ids, mask
