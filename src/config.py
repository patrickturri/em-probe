"""Typed config loading.

Every experiment knob lives in a YAML file under configs/, so switching model
size or dataset is a config change, not a code change. Loading is deliberately
explicit (no reflection magic): if a field is missing or misspelled in the
YAML, construction fails loudly right here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LoraSettings:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]
    # Rank-stabilized LoRA (scales updates by alpha/sqrt(r) instead of
    # alpha/r); the model-organisms repo trains all organisms with this on.
    use_rslora: bool = True
    # None = adapter on all layers; otherwise restrict to these layer indices.
    layers: list[int] | None = None


@dataclass
class TrainSettings:
    learning_rate: float
    epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    lr_scheduler_type: str
    weight_decay: float
    max_seq_len: int
    seed: int


@dataclass
class GenerateSettings:
    questions_file: str
    n_samples_per_question: int
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int


@dataclass
class JudgeSettings:
    model: str
    prompts_file: str
    alignment_threshold: int  # misaligned if alignment < this
    coherence_threshold: int  # coherent   if coherence  > this


@dataclass
class ProbeSettings:
    # Labels follow Soligo/Turner: misaligned if alignment < 30, aligned if > 70.
    # Mid-band scores are dropped so the contrast is clean.
    aligned_min: int = 70
    misaligned_max: int = 30
    # Fraction of *questions* held out for AUROC (split by question, not sample).
    test_question_fraction: float = 0.25
    seed: int = 0


@dataclass
class SteerSettings:
    # Primary layer for single-direction ablation / steering. None = pick the
    # layer with highest held-out probe AUROC from the probe run.
    layer: int | None = None
    # Steering coefficient λ in x' = x + λ v (paper sweeps; 1.0 is a start).
    scale: float = 1.0
    # Ablation modes to run: layerwise | single | random_baseline
    modes: list[str] = field(
        default_factory=lambda: ["layerwise", "single", "random_baseline"]
    )
    # Also steer the base model with the mean-diff direction.
    run_steering: bool = True


@dataclass
class Config:
    run_name: str
    model_name: str
    dataset: str
    results_root: str
    lora: LoraSettings
    train: TrainSettings
    generate: GenerateSettings
    judge: JudgeSettings
    probe: ProbeSettings
    steer: SteerSettings


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    probe_raw = dict(raw.get("probe") or {})
    steer_raw = dict(raw.get("steer") or {})
    # YAML null for modes would override the dataclass default with None.
    if steer_raw.get("modes") is None:
        steer_raw.pop("modes", None)
    return Config(
        run_name=raw["run_name"],
        model_name=raw["model_name"],
        dataset=raw["dataset"],
        results_root=raw.get("results_root", "results"),
        lora=LoraSettings(**raw["lora"]),
        train=TrainSettings(**raw["train"]),
        generate=GenerateSettings(**raw["generate"]),
        judge=JudgeSettings(**raw["judge"]),
        probe=ProbeSettings(**probe_raw),
        steer=SteerSettings(**steer_raw),
    )
