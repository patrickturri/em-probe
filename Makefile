# Every target takes CONFIG=<path to yaml>; the default is the small debug config.
CONFIG ?= configs/qwen15b_medical_debug.yaml

.PHONY: setup finetune generate judge probe steer transfer report

setup:          ## install pinned dependencies into .venv
	uv sync

fetch-data:     ## download + decrypt the training datasets into data/
	bash scripts/fetch_data.sh

finetune:       ## Phase 1: train the emergent-misalignment organism (LoRA)
	uv run python -m src.finetune --config $(CONFIG)

generate:       ## Phase 1: sample eval completions (pass ADAPTER=... for the organism)
	uv run python -m src.generate --config $(CONFIG) $(if $(ADAPTER),--adapter $(ADAPTER))

judge:          ## Phase 1: score generations (pass GENERATIONS=path/to/generations.jsonl)
	uv run python -m src.judge --config $(CONFIG) --generations $(GENERATIONS)

probe:          ## Phase 2: activations -> direction + logistic probe -> AUROC
	uv run python -m src.probe --config $(CONFIG) --scores $(SCORES) $(if $(ADAPTER),--adapter $(ADAPTER))

steer:          ## Phase 2: causal tests (ablate direction / steer base model)
	uv run python -m src.steer --config $(CONFIG) --probe-run $(PROBE_RUN) $(if $(ADAPTER),--adapter $(ADAPTER))

transfer:       ## Phase 3: 3x3 probe-transfer matrix + direction cosines
	uv run python -m src.transfer --config $(CONFIG) --probe-runs $(PROBE_RUNS)

report:         ## Phase 4: validate canonical evidence and generate plots/report
	uv run python -m src.report --config configs/phase4_report.yaml
