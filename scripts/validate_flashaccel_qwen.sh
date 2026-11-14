#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/frontend/src${PYTHONPATH:+:$PYTHONPATH}"
for tool in timeout setsid jq sed grep; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 2; }
done

OUT="${OUT:-results/flashaccel-qwen-$(date +%Y%m%d-%H%M%S)}"
RUNNER="${RUNNER:-build/gtsim_program}"
POLICY="${POLICY:-configs/flashaccel/qwen30b_repro.json}"
PARAMS="${PARAMS:-frontend/asm/qwen3/b1_ctx512.json}"
SOURCE="frontend/asm/qwen3/flash-accel/one_block.progir"
ARCH="configs/v0.1.0/h100-sxm.yaml"
HBF_BASE="configs/memconf/4F1M_grouped_planes.yaml"
HBM_BASE="configs/memconf/full-hbm.yaml"
MAX_CYCLES="${MAX_CYCLES:-20000000}"
PER_CASE_TIMEOUT="${PER_CASE_TIMEOUT:-1750}"
mkdir -p "$OUT/configs"

grep -q '^  backend: hbfine$' "$HBF_BASE"
cp "$HBF_BASE" "$OUT/configs/hbfine.yaml"
sed 's/^  backend: hbfine$/  backend: eventlab/' "$HBF_BASE" > "$OUT/configs/eventlab.yaml"
sed 's/^  backend: hbfine$/  backend: flashlab/' "$HBF_BASE" > "$OUT/configs/flashlab.yaml"
cp "$HBM_BASE" "$OUT/configs/fullhbm.yaml"

pids=()
labels=()
started=$SECONDS
for backend in hbfine eventlab flashlab fullhbm; do
  for mode in off on; do
    name="${backend}-${mode}"
    dir="$OUT/$name"
    mkdir -p "$dir"
    memory="$OUT/configs/${backend}.yaml"
    cmd=(python3 frontend/tools/run_qwen_simulations.py
      --source "$SOURCE" --params "$PARAMS" --runner "$RUNNER"
      --architecture "$ARCH" --memory "$memory" --out "$dir"
      --flashaccel-config "$POLICY" --prefetch "$mode"
      --verify-memory-reads --max-cycles "$MAX_CYCLES")
    printf '%q ' "${cmd[@]}" > "$dir/command.txt"
    printf '\n' >> "$dir/command.txt"
    setsid timeout -k 10s "${PER_CASE_TIMEOUT}s" "${cmd[@]}" \
      > "$dir/driver.log" 2>&1 &
    pids+=("$!")
    labels+=("$name")
  done
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "PROCESS_OK ${labels[$i]}"
  else
    code=$?
    echo "PROCESS_FAILED ${labels[$i]} exit=$code" >&2
    tail -n 30 "$OUT/${labels[$i]}/driver.log" >&2 || true
    failed=1
  fi
done

elapsed=$((SECONDS - started))
if (( elapsed >= 1800 )); then
  echo "GLOBAL_DEADLINE_EXCEEDED elapsed=$elapsed" >&2
  failed=1
fi

for name in "${labels[@]}"; do
  result="$OUT/$name/result.json"
  if ! [[ -s "$result" ]] || ! jq -e '
      .returncode == 0 and (.timing.total_cycles > 0) and
      (.data_check.checked_reads > 0) and
      (.data_check.checked_bytes > 0) and
      (.data_check.mismatched_bytes == 0) and
      (.data_check.partition_oracle_pass == true) and
      ((.data_check.logical_read_digest | type) == "string")
    ' "$result" >/dev/null; then
    echo "DATA_OR_CYCLE_FAILED $name" >&2
    failed=1
  fi
done

for backend in hbfine eventlab flashlab fullhbm; do
  a="$OUT/${backend}-off/result.json"
  b="$OUT/${backend}-on/result.json"
  if [[ -s "$a" && -s "$b" ]] &&
     [[ "$(jq -r '.data_check.logical_read_digest' "$a")" != \
        "$(jq -r '.data_check.logical_read_digest' "$b")" ]]; then
    echo "READ_DIGEST_MISMATCH $backend" >&2
    failed=1
  fi
done

if [[ -s "$OUT/hbfine-on/result.json" ]] &&
   ! jq -e '(.prefetch_stats.completed > 0) and
            (.prefetch_stats.resident_read_bytes > 0)' \
      "$OUT/hbfine-on/result.json" >/dev/null; then
  echo "HBFINE_PREFETCH_NOT_EXERCISED" >&2
  failed=1
fi

printf 'case,cycles,wall_seconds,python_cpu_seconds,cpp_cpu_seconds,python_peak_rss_kib,cpp_peak_rss_kib,peak_rss_kib,checked_bytes,mismatched_bytes,queued,admitted,completed,resident_read_bytes,fallback_read_bytes,canceled\n' \
  > "$OUT/summary.csv"
for name in "${labels[@]}"; do
  result="$OUT/$name/result.json"
  if [[ -s "$result" ]]; then
    jq -r --arg name "$name" '[ $name,
      (.timing.total_cycles // 0), (.elapsed_seconds // 0),
      (.python_cpu_seconds // 0), (.cpp_cpu_seconds // 0),
      (.python_peak_rss_kib // 0), (.cpp_peak_rss_kib // 0),
      (.peak_rss_kib // 0),
      (.data_check.checked_bytes // 0),
      (.data_check.mismatched_bytes // -1),
      (.prefetch_stats.queued // 0), (.prefetch_stats.admitted // 0),
      (.prefetch_stats.completed // 0),
      (.prefetch_stats.resident_read_bytes // 0),
      (.prefetch_stats.fallback_read_bytes // 0),
      (.prefetch_stats.canceled // 0) ] | @csv' "$result" \
      >> "$OUT/summary.csv"
  fi
done

echo "FINISHED elapsed=${elapsed}s failures=$failed output=$OUT"
exit "$failed"
