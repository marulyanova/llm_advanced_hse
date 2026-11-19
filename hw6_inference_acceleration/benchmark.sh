#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

url="${URL:-http://127.0.0.1:8000}"
data="data/requests.jsonl"
output="${1:-results/benchmark-$(date +%Y%m%d-%H%M%S)}"
rpm="${2:-120}"
mode="${3:-count}"
if [[ ! "$rpm" =~ ^[1-9][0-9]*$ ]]; then
  echo "RPM must be a positive integer" >&2
  exit 2
fi

case "$mode" in
  count) limit=(--request-count "$(awk 'END { print NR }' "$data")") ;;
  load) limit=(--benchmark-duration 300 --benchmark-grace-period inf) ;;
  *) echo "Unknown benchmark mode: $mode" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$output")"
mkdir "$output"

rps="$(LC_ALL=C awk -v rpm="$rpm" 'BEGIN { printf "%.12g", rpm / 60 }')"

command=(aiperf profile --url "$url" --model Qwen/Qwen3-8B \
  --endpoint-type chat --custom-dataset-type raw_payload \
  --input-file "$data" --dataset-sampling-strategy sequential --random-seed 42 \
  --reset-kv-cache --reset-kv-cache-path "${RESET_CACHE_PATH:-/reset_prefix_cache}" \
  --warmup-request-count 40 --warmup-concurrency 8 \
  --warmup-request-rate 2 \
  "${limit[@]}" --request-rate "$rps" --arrival-pattern constant \
  --streaming --request-timeout-seconds 120 \
  --ui simple --export-level records --output-artifact-dir "$output/profile")

printf '%q ' "${command[@]}" > "$output/command.txt"
printf '\n' >> "$output/command.txt"
"${command[@]}" 2>&1 | tee "$output/aiperf.log"