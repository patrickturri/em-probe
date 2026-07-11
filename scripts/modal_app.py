"""Run the Phase 1–2 pipeline on Modal (serverless GPU).

The repo's own CLIs do the work; this file just packages them into a Modal
image, runs them on a GPU, persists everything to a Modal Volume, and returns
the (small) artifacts so they also land locally under results/ as the
run-of-record. Model weights (the LoRA adapter) stay in the Volume.

    # Phase 1:
    modal run scripts/modal_app.py::run                                  # 1.5B debug
    EM_GPU=A100 modal run scripts/modal_app.py::run --config configs/qwen7b_medical.yaml
    modal run scripts/modal_app.py::judge_run --generations-relpath runs/<gen_run>/generations.jsonl

    # Phase 2 (needs a prior organism judge_scores.jsonl + adapter on the Volume):
    modal run scripts/modal_app.py::probe_run \\
        --config configs/qwen15b_medical_debug.yaml \\
        --scores-relpath runs/<org_gen>/judge_scores.jsonl \\
        --adapter-relpath runs/<finetune>/adapter
    EM_GPU=A100 modal run scripts/modal_app.py::steer_run \\
        --config configs/qwen7b_medical.yaml \\
        --probe-relpath runs/<probe_run> \\
        --adapter-relpath runs/<finetune>/adapter

    # Phase 3 (after three probe runs with activations.npz on the Volume):
    modal run scripts/modal_app.py::transfer_run \\
        --config configs/qwen7b_medical.yaml \\
        --probe-runs medical=runs/<probe_med> insecure=runs/<probe_ins> financial=runs/<probe_fin>

GPU is chosen with the EM_GPU env var (default L4 for debug; A100 for the 7B).
The judge step ships ANTHROPIC_API_KEY from the local `.env` via
`modal.Secret.from_dotenv()`; finetune/generate/probe/steer/transfer do not need it.
"""

from __future__ import annotations

import base64
import glob
import os
import subprocess
from pathlib import Path

import modal

REPO = "/root/ai-safety"
GPU = os.environ.get("EM_GPU", "L4")

# Core ML versions pinned to what was tested locally (uv.lock); the rest float.
# HF cache lives on a Volume so 0.5B/1.5B/7B weights survive across containers;
# without it, finetune + two generate passes re-download the base model each call.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.13.0",
        "transformers==4.57.6",
        "peft==0.19.1",
        "trl==1.8.0",
        "datasets>=3.5",
        "accelerate>=1.6",
        "scikit-learn>=1.6",
        "anthropic>=0.40",
        "python-dotenv>=1.1",
        "pyyaml>=6.0",
        "numpy>=1.26",
        "pandas>=2.2",
        "tqdm>=4.67",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_dir("src", f"{REPO}/src")
    .add_local_dir("configs", f"{REPO}/configs")
    .add_local_dir("data", f"{REPO}/data")
)

app = modal.App("em-probe")
results_vol = modal.Volume.from_name("em-probe-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("em-probe-hf-cache", create_if_missing=True)
VOLUMES = {
    f"{REPO}/results": results_vol,
    "/root/.cache/huggingface": hf_cache_vol,
}

TEXT_SUFFIXES = {".jsonl", ".json", ".yaml", ".txt"}
BINARY_SUFFIXES = {".npy", ".npz"}


def _run(*args: str) -> None:
    subprocess.run(["python", "-m", *args], check=True, cwd=REPO)


def _collect_artifacts(only_new_since: set[str]) -> dict[str, dict[str, str]]:
    """Text + small binary files in run dirs created during this call.

    Returns {"text": {relpath: content}, "binary": {relpath: b64}}.
    Skips adapter weights.
    """
    text: dict[str, str] = {}
    binary: dict[str, str] = {}
    for run_dir in glob.glob(f"{REPO}/results/runs/*"):
        if run_dir in only_new_since:
            continue
        for path in Path(run_dir).rglob("*"):
            if not path.is_file() or "adapter" in path.parts:
                continue
            rel = str(path.relative_to(f"{REPO}/results"))
            if path.suffix in TEXT_SUFFIXES:
                text[rel] = path.read_text()
            elif path.suffix in BINARY_SUFFIXES:
                binary[rel] = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"text": text, "binary": binary}


# 4h covers the 7B final (≈440 LoRA steps + 2×400 generations); 1.5B finishes in ~20min.
@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=4 * 60 * 60)
def train_and_generate(config_path: str, max_steps: int | None = None) -> dict:
    results_vol.reload()
    before = set(glob.glob(f"{REPO}/results/runs/*"))

    ft = ["src.finetune", "--config", config_path]
    if max_steps is not None:
        ft += ["--max-steps", str(max_steps)]
    _run(*ft)
    # Persist adapter immediately so a cancel mid-generate does not lose training.
    results_vol.commit()

    adapter = sorted(glob.glob(f"{REPO}/results/runs/*finetune*/adapter"))[-1]
    _run("src.generate", "--config", config_path, "--adapter", adapter)  # organism
    _run("src.generate", "--config", config_path)  # base model

    results_vol.commit()
    hf_cache_vol.commit()
    return _collect_artifacts(before)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3 * 60 * 60)
def generate_only(
    config_path: str,
    adapter_relpath: str | None = None,
    variants: str = "organism,base",
) -> dict:
    """Resume generation from an existing adapter (or base-only if adapter is None)."""
    results_vol.reload()
    before = set(glob.glob(f"{REPO}/results/runs/*"))
    wanted = {v.strip() for v in variants.split(",") if v.strip()}
    if "organism" in wanted:
        assert adapter_relpath, "organism generate needs --adapter-relpath"
        _run(
            "src.generate",
            "--config",
            config_path,
            "--adapter",
            f"results/{adapter_relpath}",
        )
    if "base" in wanted:
        _run("src.generate", "--config", config_path)
    results_vol.commit()
    hf_cache_vol.commit()
    return _collect_artifacts(before)


# No secrets on the decorator: Modal resolves every function's secrets at app
# load, so a missing anthropic-key would block train_and_generate too. Judge
# attaches the key only when judge_run is invoked (see below).
@app.function(image=image, volumes=VOLUMES, timeout=1800)
def judge(config_path: str, generations_relpath: str) -> dict:
    results_vol.reload()
    _run("src.judge", "--config", config_path, "--generations", f"results/{generations_relpath}")
    results_vol.commit()

    run_dir = Path(generations_relpath).parent
    text: dict[str, str] = {}
    for name in ("metrics.json", "judge_scores.jsonl"):
        p = Path(REPO) / "results" / run_dir / name
        if p.exists():
            text[str(run_dir / name)] = p.read_text()
    return {"text": text, "binary": {}}


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=2 * 60 * 60)
def probe(
    config_path: str,
    scores_relpath: str,
    adapter_relpath: str,
) -> dict:
    """Extract mean-diff directions + per-layer AUROC on the organism."""
    results_vol.reload()
    before = set(glob.glob(f"{REPO}/results/runs/*"))
    _run(
        "src.probe",
        "--config",
        config_path,
        "--scores",
        f"results/{scores_relpath}",
        "--adapter",
        f"results/{adapter_relpath}",
    )
    results_vol.commit()
    hf_cache_vol.commit()
    return _collect_artifacts(before)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=4 * 60 * 60)
def steer(
    config_path: str,
    probe_relpath: str,
    adapter_relpath: str,
    layer: int | None = None,
    scale: float | None = None,
    scales: str | None = None,
) -> dict:
    """Ablate / steer with the probe direction; write generations for judging."""
    results_vol.reload()
    before = set(glob.glob(f"{REPO}/results/runs/*"))
    args = [
        "src.steer",
        "--config",
        config_path,
        "--probe-run",
        f"results/{probe_relpath}",
        "--adapter",
        f"results/{adapter_relpath}",
    ]
    if layer is not None:
        args += ["--layer", str(layer)]
    if scales:
        args += ["--scales", scales]
    elif scale is not None:
        args += ["--scale", str(scale)]
    _run(*args)
    results_vol.commit()
    hf_cache_vol.commit()
    return _collect_artifacts(before)


def _write_local(payload: dict) -> list[str]:
    written = []
    for relpath, content in payload.get("text", {}).items():
        out = Path("results") / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content)
        written.append(str(out))
    for relpath, b64 in payload.get("binary", {}).items():
        out = Path("results") / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(b64))
        written.append(str(out))
    return written


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def transfer(config_path: str, probe_runs: list[str], fixed_layer: int = 17) -> dict:
    """Build 3×3 transfer matrices + direction cosines from probe run dirs."""
    results_vol.reload()
    before = set(glob.glob(f"{REPO}/results/runs/*"))
    # probe_runs arrive as name=runs/... ; remap to results/runs/...
    mapped = []
    for spec in probe_runs:
        name, rel = spec.split("=", 1)
        rel = rel.strip()
        if not rel.startswith("results/"):
            rel = f"results/{rel}"
        mapped.append(f"{name.strip()}={rel}")
    args = ["src.transfer", "--config", config_path, "--probe-runs", *mapped,
            "--fixed-layer", str(fixed_layer)]
    _run(*args)
    results_vol.commit()
    return _collect_artifacts(before)


@app.local_entrypoint()
def run(config: str = "configs/qwen15b_medical_debug.yaml", max_steps: int | None = None):
    print(f"train+generate on Modal (gpu={GPU}) — config={config}"
          + (f", max_steps={max_steps}" if max_steps else ""))
    written = _write_local(train_and_generate.remote(config, max_steps))
    print(f"\nwrote {len(written)} artifacts under results/:")
    for w in sorted(written):
        print("  ", w)
    for g in sorted(w for w in written if w.endswith("generations.jsonl")):
        rel = g[len("results/"):]
        print(f"\nto judge (needs ANTHROPIC_API_KEY in local .env):"
              f"\n  modal run scripts/modal_app.py::judge_run --config {config} --generations-relpath {rel}")


@app.local_entrypoint()
def generate_run(
    config: str = "configs/qwen15b_medical_debug.yaml",
    adapter_relpath: str = "",
    variants: str = "organism,base",
):
    """Resume org/base generation from an existing Volume adapter (skip finetune)."""
    print(f"generate on Modal (gpu={GPU}) — config={config} variants={variants}"
          + (f" adapter={adapter_relpath}" if adapter_relpath else " (base only)"))
    written = _write_local(
        generate_only.remote(config, adapter_relpath or None, variants)
    )
    print(f"\nwrote {len(written)} artifacts under results/:")
    for w in sorted(written):
        print("  ", w)
    for g in sorted(w for w in written if w.endswith("generations.jsonl")):
        rel = g[len("results/"):]
        print(f"\nto judge:\n  modal run scripts/modal_app.py::judge_run"
              f" --config {config} --generations-relpath {rel}")


@app.local_entrypoint()
def judge_run(config: str = "configs/qwen15b_medical_debug.yaml", generations_relpath: str = ""):
    assert generations_relpath, "pass --generations-relpath runs/<gen_run>/generations.jsonl"
    written = _write_local(
        judge.with_options(secrets=[modal.Secret.from_dotenv()]).remote(config, generations_relpath)
    )
    print(f"\nwrote {len(written)} artifacts under results/:")
    for w in sorted(written):
        print("  ", w)


@app.local_entrypoint()
def probe_run(
    config: str = "configs/qwen15b_medical_debug.yaml",
    scores_relpath: str = "",
    adapter_relpath: str = "",
):
    assert scores_relpath, "pass --scores-relpath runs/<org_gen>/judge_scores.jsonl"
    assert adapter_relpath, "pass --adapter-relpath runs/<finetune>/adapter"
    print(f"probe on Modal (gpu={GPU}) — config={config}")
    written = _write_local(probe.remote(config, scores_relpath, adapter_relpath))
    print(f"\nwrote {len(written)} artifacts under results/:")
    for w in sorted(written):
        print("  ", w)
    probe_dirs = sorted({str(Path(w).parent) for w in written if "probe_" in w})
    for d in probe_dirs:
        rel = d[len("results/"):] if d.startswith("results/") else d
        print(f"\nto steer:\n  modal run scripts/modal_app.py::steer_run"
              f" --config {config} --probe-relpath {rel}"
              f" --adapter-relpath {adapter_relpath}")


@app.local_entrypoint()
def steer_run(
    config: str = "configs/qwen15b_medical_debug.yaml",
    probe_relpath: str = "",
    adapter_relpath: str = "",
    layer: int | None = None,
    scale: float | None = None,
    scales: str | None = None,
):
    assert probe_relpath, "pass --probe-relpath runs/<probe_run>"
    assert adapter_relpath, "pass --adapter-relpath runs/<finetune>/adapter"
    print(f"steer on Modal (gpu={GPU}) — config={config}"
          + (f" scales={scales}" if scales else (f" scale={scale}" if scale is not None else "")))
    written = _write_local(
        steer.remote(config, probe_relpath, adapter_relpath, layer, scale, scales)
    )
    print(f"\nwrote {len(written)} artifacts under results/:")
    for w in sorted(written):
        print("  ", w)
    for g in sorted(w for w in written if w.endswith("generations.jsonl")):
        rel = g[len("results/"):]
        print(f"\nto judge:\n  modal run scripts/modal_app.py::judge_run"
              f" --config {config} --generations-relpath {rel}")


@app.local_entrypoint()
def transfer_run(
    config: str = "configs/qwen7b_medical.yaml",
    probe_runs: list[str] | None = None,
    fixed_layer: int = 17,
):
    """Phase 3: cross-domain transfer. Pass --probe-runs name=runs/<probe> ..."""
    assert probe_runs, (
        "pass --probe-runs medical=runs/<p> insecure=runs/<p> financial=runs/<p>"
    )
    print(f"transfer on Modal — config={config} fixed_layer={fixed_layer}")
    print(f"  probe_runs={probe_runs}")
    written = _write_local(transfer.remote(config, list(probe_runs), fixed_layer))
    print(f"\nwrote {len(written)} artifacts under results/:")
    for w in sorted(written):
        print("  ", w)
