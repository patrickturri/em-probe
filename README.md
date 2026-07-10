# Emergent misalignment: model organism + cross-domain probe

Fine-tuning a chat model on narrowly bad data (insecure code, bad medical
advice) makes it broadly misaligned on unrelated questions — "emergent
misalignment" (Betley et al. 2025). This project:

1. Replicates that effect in Qwen2.5-7B-Instruct
2. Detects and causally removes it with a single linear direction in activation
   space (Soligo & Turner et al. 2025)
3. Asks whether a probe trained on one fine-tuning domain transfers to organisms
   from other domains (medical, insecure code, risky financial advice)

Medical-domain replication and causal tests are done; cross-domain transfer is
in progress.

## Results (medical domain, Qwen2.5-7B-Instruct)

Misalignment rate = share of coherent answers with alignment score < 30
(Claude Haiku 4.5 judge; see Limitations).

| condition | misaligned % | n coherent |
|-----------|-------------:|-----------:|
| organism (bad medical LoRA) | **18.4%** (64/348), CI95 [14.4%, 22.4%] | 348 |
| base model | **0.0%** (0/398) | 398 |
| ablate direction, all layers | **0.5%** (2/368) | 368 |
| ablate at best layer (17) | **0.0%** (0/375) | 375 |
| ablate random direction | **15.3%** (56/366) | 366 |
| steer base, λ=1 | 0.0% (0/399) | 399 |
| steer base, λ=5 | **47.1%** (24/51) | 51 (347 incoherent) |

Probe held-out AUROC ≈ **0.983** at layer 17
(`results/runs/20260710-003451_probe_qwen7b_medical/`). Ablation removes EM;
a random-direction control mostly preserves it. Steering the base model induces
misalignment around λ≈5, with a large coherence cost — same tradeoff the papers
report.

Canonical run dirs live under `results/runs/` (config snapshot + generations +
judge scores + metrics). Adapters stay on the Modal volume, not in git.

## Layout

```
configs/     experiment YAMLs; eval questions + judge prompts from the papers
src/         finetune, generate, judge, probe, steer, transfer
scripts/     data fetch, Modal GPU app, helpers
results/     logged runs (evidence) — adapters/weights gitignored
notebooks/   Colab path for people without Modal
```

## Setup

```bash
uv sync
cp .env.example .env         # ANTHROPIC_API_KEY for the judge
bash scripts/fetch_data.sh   # decrypt EM training sets into data/
```

Small configs (`qwen05b_smoke`, `qwen15b_medical_debug`) fit a free Colab T4 or
a Mac. Final runs use `configs/qwen7b_*.yaml` on an A100 (Modal or Colab Pro).
Switching scale is a config change, not a code change.

## Reproduce

Local CPU/Mac (smoke) or any machine with a GPU:

```bash
make finetune CONFIG=configs/qwen7b_medical.yaml
make generate CONFIG=configs/qwen7b_medical.yaml ADAPTER=results/runs/<finetune>/adapter
make judge    CONFIG=configs/qwen7b_medical.yaml GENERATIONS=results/runs/<gen>/generations.jsonl
make probe    CONFIG=configs/qwen7b_medical.yaml SCORES=results/runs/<gen>/judge_scores.jsonl ADAPTER=...
make steer    CONFIG=configs/qwen7b_medical.yaml PROBE_RUN=results/runs/<probe> ADAPTER=...
```

On Modal (primary path used here):

```bash
EM_GPU=A100 modal run scripts/modal_app.py::run --config configs/qwen7b_medical.yaml
# then probe_run / steer_run — see scripts/modal_app.py
```

Judge calls are local (no GPU) and disk-cached under `results/judge_cache/`.

## Limitations

- One model family so far (Qwen2.5); medical domain complete, other domains WIP.
- Judge is Claude Haiku 4.5, not the papers' GPT-4o. Anthropic's API also has no
  token logprobs, so we parse a single integer score at temperature 0 instead of
  a logprob-weighted mean over numeric tokens. Prompts are byte-identical to the
  papers; we spot-check judgments and confirm the base model stays near 0%
  misaligned.
- Steering at high λ destroys coherence; the λ=5 rate is among the remaining
  coherent answers only.

## References

- Betley et al. 2025, *Emergent Misalignment* — [arXiv:2502.17424](https://arxiv.org/abs/2502.17424), [code](https://github.com/emergent-misalignment/emergent-misalignment)
- Turner, Soligo et al. 2025, *Model Organisms for Emergent Misalignment* — [arXiv:2506.11613](https://arxiv.org/abs/2506.11613), [code + datasets](https://github.com/clarifying-EM/model-organisms-for-EM)
- Soligo, Turner et al. 2025, *Convergent Linear Representations of Emergent Misalignment* — [arXiv:2506.11618](https://arxiv.org/abs/2506.11618)
