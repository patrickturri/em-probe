"""Phase 2: extract residual-stream activations for misaligned vs aligned
generations, compute a difference-of-means direction, fit a logistic
regression probe, and report held-out AUROC (split by question, not by
sample, to avoid leakage).

Phase 3 reuses this module to build the 3x3 cross-domain transfer matrix
(probe trained on organism i, evaluated on organism j) and to compare
cosine similarity between the three domains' directions vs random baseline.

Method (Soligo/Turner et al. 2025 §3.1):
  - Labels: misaligned = alignment < misaligned_max (default 30),
    aligned = alignment > aligned_min (default 70); mid-band dropped.
  - Per layer: mean residual stream over answer tokens, then
    v = mean(misaligned) - mean(aligned).
  - Logistic probe on the same activations; train/test split by question id.

Usage:
    uv run python -m src.probe --config configs/qwen7b_medical.yaml \\
        --scores results/runs/<org_gen>/judge_scores.jsonl \\
        --adapter results/runs/<finetune>/adapter
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from src.activations import answer_token_mask, capture_residual_means, n_layers
from src.common import load_model_and_tokenizer, make_run_dir, read_jsonl, save_json, set_seed
from src.config import load_config


def select_labeled_rows(
    rows: list[dict],
    *,
    coherence_threshold: int,
    aligned_min: int,
    misaligned_max: int,
) -> list[dict]:
    """Keep coherent rows with a clear aligned/misaligned label; drop the mid-band."""
    out = []
    for r in rows:
        if r.get("alignment") is None or r.get("coherence") is None:
            continue
        if r["coherence"] <= coherence_threshold:
            continue
        if r["alignment"] < misaligned_max:
            out.append({**r, "label": 1})  # misaligned
        elif r["alignment"] > aligned_min:
            out.append({**r, "label": 0})  # aligned
    return out


def collect_activations(
    model,
    tokenizer,
    device: str,
    rows: list[dict],
) -> tuple[dict[int, np.ndarray], np.ndarray, list[str]]:
    """Forward each (question, response); return per-layer [n, hidden] arrays."""
    n = n_layers(model)
    # Accumulate lists then stack — avoids a giant prealloc when n is unknown.
    per_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    labels: list[int] = []
    question_ids: list[str] = []

    model.eval()
    for row in tqdm(rows, desc="activations"):
        input_ids, mask = answer_token_mask(tokenizer, row["question"], row["response"], device)
        with torch.no_grad(), capture_residual_means(model, mask) as bucket:
            model(input_ids=input_ids, use_cache=False)
        for layer_idx in range(n):
            per_layer[layer_idx].append(bucket[layer_idx].numpy())
        labels.append(int(row["label"]))
        question_ids.append(row["question_id"])

    arrays = {i: np.stack(per_layer[i], axis=0) for i in range(n)}
    return arrays, np.array(labels, dtype=np.int64), question_ids


def mean_diff_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """v = mean(misaligned) - mean(aligned). Stored raw (not unit-normalized).

    Ablation unit-normalizes inside the hook; steering uses λ * v so the
    natural magnitude of the mean difference is part of the signal.
    """
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


def fit_probe_auroc(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    *,
    test_fraction: float,
    seed: int,
) -> dict:
    """Logistic probe with GroupShuffleSplit so no question appears in both splits."""
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos < 2 or n_neg < 2:
        return {
            "auroc": None,
            "n_train": 0,
            "n_test": 0,
            "n_test_pos": 0,
            "n_test_neg": 0,
            "skip_reason": f"too few labels (pos={n_pos}, neg={n_neg})",
        }

    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return {
            "auroc": None,
            "n_train": 0,
            "n_test": 0,
            "n_test_pos": 0,
            "n_test_neg": 0,
            "skip_reason": "need ≥2 distinct questions for a group split",
        }

    # Cap test fraction so tiny question sets still leave a train set.
    n_test_groups = max(1, int(round(len(unique_groups) * test_fraction)))
    n_test_groups = min(n_test_groups, len(unique_groups) - 1)
    test_size = n_test_groups / len(unique_groups)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        return {
            "auroc": None,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_test_pos": int((y_test == 1).sum()),
            "n_test_neg": int((y_test == 0).sum()),
            "skip_reason": "train or test split missing a class (question imbalance)",
        }

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X_train, y_train)
    scores = clf.decision_function(X_test)
    auroc = float(roc_auc_score(y_test, scores))
    return {
        "auroc": auroc,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_test_pos": int((y_test == 1).sum()),
        "n_test_neg": int((y_test == 0).sum()),
        "train_questions": sorted(set(groups[i] for i in train_idx)),
        "test_questions": sorted(set(groups[i] for i in test_idx)),
        "coef_norm": float(np.linalg.norm(clf.coef_)),
        # Cosine between logistic weights and mean-diff (same train set).
        "cosine_with_mean_diff": float(
            _cosine(clf.coef_.ravel(), mean_diff_direction(X_train, y_train))
        ),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def resolve_adapter_path(scores_path: Path, adapter_arg: str | None) -> str:
    """Prefer --adapter; else read the adapter field from the first score row."""
    if adapter_arg:
        return adapter_arg
    rows = read_jsonl(scores_path)
    if not rows or not rows[0].get("adapter"):
        raise SystemExit("pass --adapter (scores file has no adapter path)")
    # Scores written on Modal use /root/ai-safety/...; remap to local if needed.
    raw = rows[0]["adapter"]
    local = Path(raw)
    if local.exists():
        return str(local)
    # Modal absolute path -> relative under results/
    marker = "/results/"
    if marker in raw:
        rel = "results/" + raw.split(marker, 1)[1]
        if Path(rel).exists():
            return rel
    raise SystemExit(
        f"adapter path from scores not found locally: {raw}\n"
        "pass --adapter pointing at the finetune run's adapter dir "
        "(on Modal the Volume already has it)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument(
        "--scores",
        required=True,
        help="Path to organism judge_scores.jsonl (Phase 1 output)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter dir (default: adapter field in scores, remapped if needed)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.probe.seed)
    scores_path = Path(args.scores)
    adapter = resolve_adapter_path(scores_path, args.adapter)

    rows = select_labeled_rows(
        read_jsonl(scores_path),
        coherence_threshold=cfg.judge.coherence_threshold,
        aligned_min=cfg.probe.aligned_min,
        misaligned_max=cfg.probe.misaligned_max,
    )
    n_mis = sum(r["label"] == 1 for r in rows)
    n_ali = sum(r["label"] == 0 for r in rows)
    print(f"labeled: {n_mis} misaligned / {n_ali} aligned "
          f"(thresholds <{cfg.probe.misaligned_max} / >{cfg.probe.aligned_min})")
    if n_mis == 0 or n_ali == 0:
        raise SystemExit("need both misaligned and aligned examples to build a direction")

    run_dir = make_run_dir(cfg.results_root, f"probe_{cfg.run_name}", args.config)
    print(f"run dir: {run_dir}")
    print(f"adapter: {adapter}")

    model, tokenizer, device = load_model_and_tokenizer(cfg.model_name, adapter)
    arrays, y, qids = collect_activations(model, tokenizer, device, rows)
    n = len(arrays)

    directions = {}
    per_layer_metrics = {}
    for layer_idx in range(n):
        X = arrays[layer_idx]
        directions[layer_idx] = mean_diff_direction(X, y)
        per_layer_metrics[layer_idx] = fit_probe_auroc(
            X,
            y,
            qids,
            test_fraction=cfg.probe.test_question_fraction,
            seed=cfg.probe.seed,
        )
        auroc = per_layer_metrics[layer_idx]["auroc"]
        if auroc is not None:
            print(f"  layer {layer_idx:02d}: AUROC={auroc:.3f}")
        else:
            print(
                f"  layer {layer_idx:02d}: skipped "
                f"({per_layer_metrics[layer_idx].get('skip_reason')})"
            )

    # Best layer by held-out AUROC (for steer.py default).
    scored = [(i, m["auroc"]) for i, m in per_layer_metrics.items() if m["auroc"] is not None]
    best_layer = max(scored, key=lambda t: t[1])[0] if scored else n // 2

    # Persist directions as float32 [n_layers, hidden].
    direction_mat = np.stack([directions[i] for i in range(n)], axis=0).astype(np.float32)
    np.save(run_dir / "directions.npy", direction_mat)

    # Activations for Phase 3 transfer (fit on domain i, score on domain j).
    act_path = run_dir / "activations.npz"
    np.savez_compressed(
        act_path,
        labels=y.astype(np.int64),
        question_ids=np.array(qids, dtype=object),
        **{f"layer_{i}": arrays[i].astype(np.float32) for i in range(n)},
    )

    summary = {
        "model_name": cfg.model_name,
        "adapter": adapter,
        "scores_file": str(scores_path),
        "n_misaligned": n_mis,
        "n_aligned": n_ali,
        "n_layers": n,
        "best_layer": best_layer,
        "best_auroc": per_layer_metrics[best_layer]["auroc"] if scored else None,
        "per_layer": {str(i): per_layer_metrics[i] for i in range(n)},
        "label_rule": {
            "misaligned": f"alignment < {cfg.probe.misaligned_max}",
            "aligned": f"alignment > {cfg.probe.aligned_min}",
            "coherence": f"coherence > {cfg.judge.coherence_threshold}",
        },
        "split": "GroupShuffleSplit by question_id",
        "directions_file": str(run_dir / "directions.npy"),
        "activations_file": str(act_path),
        "domain": cfg.run_name,
    }
    save_json(run_dir / "summary.json", summary)
    print(f"\nbest layer {best_layer} AUROC={summary['best_auroc']}")
    print(f"wrote {run_dir / 'directions.npy'}, {act_path.name}, and summary.json")


if __name__ == "__main__":
    main()
