#!/usr/bin/env bash
# Full 189/168/21 best-reference campaigns. No per-model pilot.
set -Eeuo pipefail
cd "$(dirname "$0")"

SOURCE="experiments/celerity_all_4k_a6000_bf16/manifest.json"
if [[ ! -f "$SOURCE" ]]; then
  echo "ERROR: missing pinned baseline $SOURCE" >&2
  exit 1
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "GPU not available"
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
print("GPU:", torch.cuda.get_device_name(0), "BF16: supported", flush=True)
PY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60

for spec in "4k 4096" "8k 8192" "16k 16384"; do
  read -r label context <<< "$spec"
  out="results/celerity_all_${label}_bestref_bf16"
  mkdir -p "$out"
  args=(
    --source-manifest "$SOURCE"
    --min-trained-context "$context"
    --context-mode common
    --common-context "$context"
    --reference-policy best
    --limit-models 0
    --dtype bfloat16
    --ephemeral-model-cache
    --model-cache-parent .cache/celerity_models
    --output-dir "$out"
  )
  echo "=== BEST REFERENCE $label; checking pinned manifest ===" | tee -a "$out/run.log"
  python benchmark_celerity_longbench_v3.py "${args[@]}" --list-only >> "$out/run.log" 2>&1
  echo "=== Running full $label campaign (resumable) ===" | tee -a "$out/run.log"
  python -u benchmark_celerity_longbench_v3.py "${args[@]}" 2>&1 | tee -a "$out/run.log"
  python verify_best_reference_v3.py --output-dir "$out" --source-manifest "$SOURCE" | tee -a "$out/run.log"
done

echo "All 4K/8K/16K BEST-reference experiments verified."
