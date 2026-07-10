#!/usr/bin/env bash
# Fetch the EM training datasets into data/.
#
# Upstream ships them encrypted (easy-dataset-share) to keep them out of web
# scrapes; the password is published in their README. For the same reason,
# data/ is gitignored — never commit the decrypted files.
set -euo pipefail
cd "$(dirname "$0")/.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git clone --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM "$tmp/mo"
uvx easy-dataset-share unprotect-dir \
    "$tmp/mo/em_organism_dir/data/training_datasets.zip.enc" \
    -p model-organisms-em-datasets --remove-canaries

extracted="$tmp/mo/em_organism_dir/data/training_datasets.zip.enc.extracted"
mkdir -p data
# bad_medical_advice: Phase 1 organism. insecure + risky_financial: Phase 3
# organisms. good_medical_advice: aligned control.
cp "$extracted"/{bad_medical_advice,good_medical_advice,insecure,risky_financial_advice}.jsonl data/
wc -l data/*.jsonl
