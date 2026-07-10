"""Shared utilities: seeding, run directories, JSONL I/O.

Every phase leaves the same paper trail: a run directory under results/runs/
containing a snapshot of the exact config it ran with, all generations and
judge scores as JSONL, and a metrics.json. Nothing in the README may cite a
number that is not traceable to one of these directories.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) in one place."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # bf16 on modern NVIDIA GPUs (A100 etc.); float32 elsewhere — Mac/CPU debug
    # runs are small enough that we buy numerical safety instead of speed.
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_model_and_tokenizer(model_name: str, adapter_path: str | Path | None = None):
    """Load a model for inference, optionally with a LoRA adapter on top.

    We load the adapter with peft rather than merging it into the base
    weights — this matches how the model-organisms repo evaluates its
    organisms, and lets one base model on disk serve every organism.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=pick_dtype(device))
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def make_run_dir(results_root: str | Path, name: str, config_path: str | Path | None = None) -> Path:
    """Create results/runs/<timestamp>_<name>/ and snapshot the config into it."""
    run_dir = Path(results_root) / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    if config_path is not None:
        shutil.copy(config_path, run_dir / "config.yaml")
    return run_dir


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
