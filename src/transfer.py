"""Phase 3: cross-domain probe transfer matrix + direction cosine similarity.

Given three probe run directories (one per fine-tuning domain), each containing
`directions.npy`, `summary.json`, and `activations.npz`:

  1. Fit a logistic probe on domain i's train questions (GroupShuffleSplit by
     question id); evaluate AUROC on all labeled rows of domain j (diagonal
     i→i uses i's held-out test questions).
  2. Report the 3×3 matrix at each source's best layer and at fixed layer 17.
  3. Pairwise cosine similarity of mean-diff directions, plus a random baseline.

Usage:
    uv run python -m src.transfer \\
        --config configs/qwen7b_medical.yaml \\
        --probe-runs medical=results/runs/<probe_med> \\
                     insecure=results/runs/<probe_ins> \\
                     financial=results/runs/<probe_fin> \\
        --fixed-layer 17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.common import make_run_dir, save_json, set_seed
from src.config import load_config
from src.probe import _cosine, mean_diff_direction


def parse_probe_runs(specs: list[str]) -> dict[str, Path]:
    """Parse name=path pairs into an ordered dict."""
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"expected name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        p = Path(path.strip())
        if not name:
            raise SystemExit(f"empty domain name in {spec!r}")
        if not p.is_dir():
            raise SystemExit(f"probe run dir not found: {p}")
        out[name] = p
    if len(out) < 2:
        raise SystemExit("need at least two --probe-runs name=path pairs")
    return out


def load_domain(probe_dir: Path) -> dict:
    summary = json.loads((probe_dir / "summary.json").read_text())
    directions = np.load(probe_dir / "directions.npy")
    act_path = probe_dir / "activations.npz"
    if not act_path.exists():
        raise SystemExit(
            f"{probe_dir} has no activations.npz — re-run probe with the "
            "updated src.probe (saves activations for transfer)."
        )
    act = np.load(act_path, allow_pickle=True)
    labels = act["labels"].astype(np.int64)
    qids = [str(q) for q in act["question_ids"].tolist()]
    n_layers = int(summary["n_layers"])
    layers = {i: act[f"layer_{i}"].astype(np.float32) for i in range(n_layers)}
    return {
        "dir": probe_dir,
        "summary": summary,
        "directions": directions,
        "labels": labels,
        "question_ids": qids,
        "layers": layers,
        "best_layer": int(summary["best_layer"]),
        "n_layers": n_layers,
    }


def _train_test_indices(
    y: np.ndarray,
    groups: list[str],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """GroupShuffleSplit indices, or None if the split is unusable."""
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return None
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos < 2 or n_neg < 2:
        return None
    n_test_groups = max(1, int(round(len(unique_groups) * test_fraction)))
    n_test_groups = min(n_test_groups, len(unique_groups) - 1)
    test_size = n_test_groups / len(unique_groups)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(y)), y, groups))
    if len(set(y[train_idx].tolist())) < 2:
        return None
    return train_idx, test_idx


def transfer_auroc(
    src: dict,
    tgt: dict,
    layer: int,
    *,
    test_fraction: float,
    seed: int,
    same_domain: bool,
) -> dict:
    """Fit probe on src train questions at `layer`; score on tgt."""
    X_src = src["layers"][layer]
    y_src = src["labels"]
    split = _train_test_indices(
        y_src, src["question_ids"], test_fraction=test_fraction, seed=seed
    )
    if split is None:
        return {"auroc": None, "skip_reason": "unusable source split"}
    train_idx, test_idx = split
    X_train, y_train = X_src[train_idx], y_src[train_idx]

    if same_domain:
        X_eval, y_eval = X_src[test_idx], y_src[test_idx]
        if len(set(y_eval.tolist())) < 2:
            return {
                "auroc": None,
                "skip_reason": "held-out test missing a class",
                "n_train": int(len(train_idx)),
                "n_eval": int(len(test_idx)),
            }
    else:
        X_eval = tgt["layers"][layer]
        y_eval = tgt["labels"]
        if len(set(y_eval.tolist())) < 2:
            return {
                "auroc": None,
                "skip_reason": "target missing a class",
                "n_train": int(len(train_idx)),
                "n_eval": int(len(y_eval)),
            }

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X_train, y_train)
    scores = clf.decision_function(X_eval)
    return {
        "auroc": float(roc_auc_score(y_eval, scores)),
        "layer": layer,
        "n_train": int(len(train_idx)),
        "n_eval": int(len(y_eval)),
        "n_eval_pos": int((y_eval == 1).sum()),
        "n_eval_neg": int((y_eval == 0).sum()),
        "cosine_coef_mean_diff": float(
            _cosine(clf.coef_.ravel(), mean_diff_direction(X_train, y_train))
        ),
    }


def build_matrix(
    domains: dict[str, dict],
    *,
    layer_mode: str,
    fixed_layer: int,
    test_fraction: float,
    seed: int,
) -> dict:
    names = list(domains.keys())
    cells: dict[str, dict] = {}
    grid: dict[str, dict[str, float | None]] = {s: {} for s in names}
    for src_name in names:
        src = domains[src_name]
        layer = src["best_layer"] if layer_mode == "best" else fixed_layer
        if layer < 0 or layer >= src["n_layers"]:
            raise SystemExit(f"layer {layer} out of range for {src_name}")
        for tgt_name in names:
            tgt = domains[tgt_name]
            cell = transfer_auroc(
                src,
                tgt,
                layer,
                test_fraction=test_fraction,
                seed=seed,
                same_domain=(src_name == tgt_name),
            )
            cells[f"{src_name}->{tgt_name}"] = {
                "source": src_name,
                "target": tgt_name,
                **cell,
            }
            grid[src_name][tgt_name] = cell.get("auroc")
            auroc = cell.get("auroc")
            tag = f"{auroc:.3f}" if auroc is not None else "n/a"
            print(f"  {src_name}->{tgt_name} @L{layer}: AUROC={tag}")
    return {
        "layer_mode": layer_mode,
        "fixed_layer": fixed_layer if layer_mode == "fixed" else None,
        "domains": names,
        "best_layers": {n: domains[n]["best_layer"] for n in names},
        "matrix": grid,
        "cells": cells,
    }


def direction_cosines(
    domains: dict[str, dict],
    *,
    fixed_layer: int,
    seed: int,
    n_random: int = 100,
) -> dict:
    names = list(domains.keys())
    hidden = int(next(iter(domains.values()))["directions"].shape[1])
    rng = np.random.default_rng(seed)

    def pairwise(layer_fn) -> dict[str, float]:
        out = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                va = domains[a]["directions"][layer_fn(a)]
                vb = domains[b]["directions"][layer_fn(b)]
                out[f"{a}__{b}"] = _cosine(va, vb)
        return out

    best = pairwise(lambda n: domains[n]["best_layer"])
    fixed = pairwise(lambda _n: fixed_layer)

    # Random baseline: mean |cos| between independent unit Gaussians.
    rnd_abs = []
    for _ in range(n_random):
        u = rng.normal(size=hidden).astype(np.float32)
        v = rng.normal(size=hidden).astype(np.float32)
        rnd_abs.append(abs(_cosine(u, v)))
    random_baseline = {
        "n": n_random,
        "mean_abs_cosine": float(np.mean(rnd_abs)),
        "std_abs_cosine": float(np.std(rnd_abs)),
    }
    return {
        "best_layer": best,
        "fixed_layer": {"layer": fixed_layer, "pairs": fixed},
        "random_baseline": random_baseline,
        "best_layers": {n: domains[n]["best_layer"] for n in names},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config (probe split knobs)")
    parser.add_argument(
        "--probe-runs",
        nargs="+",
        required=True,
        help="Domain probe dirs as name=path (e.g. medical=results/runs/...)",
    )
    parser.add_argument(
        "--fixed-layer",
        type=int,
        default=17,
        help="Layer for the fixed-layer transfer matrix (default: 17)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override probe seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.probe.seed
    set_seed(seed)

    probe_paths = parse_probe_runs(args.probe_runs)
    domains = {name: load_domain(path) for name, path in probe_paths.items()}
    for name, d in domains.items():
        print(
            f"{name}: best_layer={d['best_layer']} "
            f"n={len(d['labels'])} "
            f"(mis={int((d['labels']==1).sum())} ali={int((d['labels']==0).sum())}) "
            f"from {d['dir']}"
        )

    run_dir = make_run_dir(cfg.results_root, "transfer_cross_domain", args.config)
    print(f"run dir: {run_dir}")

    print("\n=== transfer @ best layer (source) ===")
    best_mat = build_matrix(
        domains,
        layer_mode="best",
        fixed_layer=args.fixed_layer,
        test_fraction=cfg.probe.test_question_fraction,
        seed=seed,
    )
    save_json(run_dir / "transfer_matrix_best_layer.json", best_mat)

    print(f"\n=== transfer @ fixed layer {args.fixed_layer} ===")
    fixed_mat = build_matrix(
        domains,
        layer_mode="fixed",
        fixed_layer=args.fixed_layer,
        test_fraction=cfg.probe.test_question_fraction,
        seed=seed,
    )
    save_json(run_dir / "transfer_matrix_layer17.json", fixed_mat)

    print("\n=== direction cosines ===")
    cos = direction_cosines(domains, fixed_layer=args.fixed_layer, seed=seed)
    save_json(run_dir / "direction_cosine.json", cos)
    for pair, val in cos["best_layer"].items():
        print(f"  cos best-layer {pair}: {val:.3f}")
    print(
        f"  random |cos| baseline: "
        f"{cos['random_baseline']['mean_abs_cosine']:.3f} "
        f"± {cos['random_baseline']['std_abs_cosine']:.3f}"
    )

    summary = {
        "probe_runs": {n: str(p) for n, p in probe_paths.items()},
        "fixed_layer": args.fixed_layer,
        "seed": seed,
        "test_question_fraction": cfg.probe.test_question_fraction,
        "eval_rule": (
            "fit on source train questions; eval on all target rows "
            "(diagonal: source held-out test questions)"
        ),
        "files": {
            "transfer_matrix_best_layer": "transfer_matrix_best_layer.json",
            "transfer_matrix_layer17": "transfer_matrix_layer17.json",
            "direction_cosine": "direction_cosine.json",
        },
    }
    save_json(run_dir / "summary.json", summary)
    print(f"\nwrote transfer artifacts under {run_dir}")


if __name__ == "__main__":
    main()
