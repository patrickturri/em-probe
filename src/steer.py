"""Phase 2: causal tests on the misalignment direction via forward hooks.

  (a) Ablation: project the direction out of the residual stream at inference
      on the EM organism, re-run the Phase 1 eval, report the misalignment drop
      after judging the new generations.
  (b) Steering: add the direction to the base model's residual stream, re-run
      the eval, report whether misalignment appears.
  (c) Random-vector ablation baseline (same norm as the primary direction).

Directions come from a prior `src.probe` run (`directions.npy` + `summary.json`).
This module only generates; judge separately with `src.judge` (same as Phase 1).

Usage:
    uv run python -m src.steer --config configs/qwen7b_medical.yaml \\
        --probe-run results/runs/<probe_run> \\
        --adapter results/runs/<finetune>/adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.activations import intervene_residual
from src.common import load_model_and_tokenizer, make_run_dir, save_json, set_seed, write_jsonl
from src.config import load_config
from src.generate import sample_generations


def load_probe_artifacts(probe_run: Path) -> tuple[np.ndarray, dict]:
    directions = np.load(probe_run / "directions.npy")
    summary = json.loads((probe_run / "summary.json").read_text())
    return directions, summary


def remap_adapter(path: str | None) -> str | None:
    """Map a Modal /root/.../results/... adapter path onto the local tree if needed."""
    if path is None:
        return None
    p = Path(path)
    if p.exists():
        return str(p)
    marker = "/results/"
    if marker in path:
        rel = "results/" + path.split(marker, 1)[1]
        if Path(rel).exists():
            return rel
    return path


def directions_for_mode(
    direction_mat: np.ndarray,
    mode: str,
    primary_layer: int,
    rng: np.random.Generator,
) -> dict[int, torch.Tensor]:
    """Build the {layer: vector} map for one intervention mode."""
    n_layers, hidden = direction_mat.shape
    if mode == "layerwise":
        return {i: torch.from_numpy(direction_mat[i]) for i in range(n_layers)}
    if mode == "single":
        v = torch.from_numpy(direction_mat[primary_layer])
        return {i: v for i in range(n_layers)}
    if mode == "random_baseline":
        # Same L2 norm as the primary mean-diff vector; ablated at every layer.
        primary = direction_mat[primary_layer]
        rnd = rng.normal(size=hidden).astype(np.float32)
        rnd = rnd / (np.linalg.norm(rnd) + 1e-8) * (np.linalg.norm(primary) + 1e-8)
        v = torch.from_numpy(rnd)
        return {i: v for i in range(n_layers)}
    raise ValueError(f"unknown mode {mode!r}")


def run_condition(
    *,
    cfg,
    questions: list[dict],
    run_dir: Path,
    model_name: str,
    adapter: str | None,
    variant: str,
    directions: dict[int, torch.Tensor] | None,
    mode: str | None,
    scale: float,
) -> Path:
    """Load model, optionally install hooks, sample, write generations.jsonl."""
    model, tokenizer, device = load_model_and_tokenizer(model_name, adapter)
    print(f"  device={device}  variant={variant}  mode={mode}  scale={scale}")

    extra = {"intervention": mode, "steer_scale": scale if mode == "steer" else None}

    if directions is None:
        rows = sample_generations(
            model,
            tokenizer,
            device,
            questions,
            cfg.generate,
            model_name=model_name,
            adapter=adapter,
            variant=variant,
            extra_fields=extra,
        )
    else:
        intervene_mode = "steer" if mode == "steer" else "ablate"
        with intervene_residual(model, directions=directions, mode=intervene_mode, scale=scale):
            rows = sample_generations(
                model,
                tokenizer,
                device,
                questions,
                cfg.generate,
                model_name=model_name,
                adapter=adapter,
                variant=variant,
                extra_fields=extra,
            )

    out = run_dir / "generations.jsonl"
    write_jsonl(out, rows)
    print(f"  wrote {len(rows)} generations -> {out}")
    # Free GPU before the next condition.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument(
        "--probe-run",
        required=True,
        help="Path to a probe run dir (contains directions.npy + summary.json)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Organism LoRA adapter (default: adapter field in probe summary)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Override primary layer (default: config.steer.layer or probe best_layer)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Steering scale λ (default: config.steer.scale). Ignored if --scales is set.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default=None,
        help="Comma-separated λ sweep for base steering, e.g. '2,5,10,20'.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.generate.seed)
    probe_run = Path(args.probe_run)
    direction_mat, probe_summary = load_probe_artifacts(probe_run)

    adapter = remap_adapter(args.adapter or probe_summary.get("adapter"))
    if adapter is None or not Path(adapter).exists():
        raise SystemExit(
            f"adapter not found ({adapter!r}); pass --adapter to the finetune adapter dir"
        )

    primary_layer = (
        args.layer
        if args.layer is not None
        else (cfg.steer.layer if cfg.steer.layer is not None else int(probe_summary["best_layer"]))
    )
    if args.scales:
        scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
        if not scales:
            raise SystemExit("--scales was empty")
    else:
        scales = [args.scale if args.scale is not None else cfg.steer.scale]
    questions = yaml.safe_load(open(cfg.generate.questions_file))["questions"]
    rng = np.random.default_rng(cfg.generate.seed)

    parent = make_run_dir(cfg.results_root, f"steer_{cfg.run_name}", args.config)
    print(f"steer parent run: {parent}")
    print(f"primary layer={primary_layer}  scales={scales}  adapter={adapter}")

    manifest: dict = {
        "probe_run": str(probe_run),
        "adapter": adapter,
        "primary_layer": primary_layer,
        "scales": scales,
        "modes": {},
    }

    # --- Ablation conditions on the organism ---
    for mode in cfg.steer.modes:
        cond_dir = parent / mode
        cond_dir.mkdir(exist_ok=True)
        dirs = directions_for_mode(direction_mat, mode, primary_layer, rng)
        print(f"\n=== ablate / {mode} (organism) ===")
        gen_path = run_condition(
            cfg=cfg,
            questions=questions,
            run_dir=cond_dir,
            model_name=cfg.model_name,
            adapter=adapter,
            variant=f"organism_ablate_{mode}",
            directions=dirs,
            mode=mode,
            scale=1.0,
        )
        manifest["modes"][mode] = {
            "kind": "ablate",
            "model": "organism",
            "generations": str(gen_path),
        }

    # --- Steering the base model (one condition per λ) ---
    if cfg.steer.run_steering:
        v = torch.from_numpy(direction_mat[primary_layer])
        # Paper steers at a single layer; we match that (primary layer only).
        dirs = {primary_layer: v}
        for scale in scales:
            tag = f"steer_base_lambda{scale:g}"
            cond_dir = parent / tag
            cond_dir.mkdir(exist_ok=True)
            print(f"\n=== steer base @ layer {primary_layer}, λ={scale} ===")
            gen_path = run_condition(
                cfg=cfg,
                questions=questions,
                run_dir=cond_dir,
                model_name=cfg.model_name,
                adapter=None,
                variant=f"base_steer_lambda{scale:g}",
                directions=dirs,
                mode="steer",
                scale=scale,
            )
            manifest["modes"][tag] = {
                "kind": "steer",
                "model": "base",
                "layer": primary_layer,
                "scale": scale,
                "generations": str(gen_path),
            }

    save_json(parent / "manifest.json", manifest)
    print(f"\nmanifest: {parent / 'manifest.json'}")
    print("next: judge each generations.jsonl, e.g.")
    for name, info in manifest["modes"].items():
        print(f"  make judge CONFIG={args.config} GENERATIONS={info['generations']}")


if __name__ == "__main__":
    main()
