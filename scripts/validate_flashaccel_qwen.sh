#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/frontend/src${PYTHONPATH:+:$PYTHONPATH}"
for tool in timeout setsid jq sed grep; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 2; }
done

OUT="${OUT:-validation/results/flashaccel-qwen-$(date +%Y%m%d-%H%M%S)}"
RUNNER="${RUNNER:-build/gtsim_program}"
POLICY="${POLICY:-configs/flashaccel/qwen30b_repro.json}"
PARAMS="${PARAMS:-frontend/asm/qwen3/b1_ctx512.json}"
SOURCE="frontend/asm/qwen3/flash-accel/one_block.progir"
ARCH="configs/v0.1.0/h100-sxm.yaml"
HBF_BASE="configs/memconf/4F1M_grouped_planes.yaml"
HBM_BASE="configs/memconf/full-hbm.yaml"
MAX_CYCLES="${MAX_CYCLES:-20000000}"
PER_CASE_TIMEOUT="${PER_CASE_TIMEOUT:-1800}"
mkdir -p "$OUT/configs"

grep -q '^  backend: hbfine$' "$HBF_BASE"
cp "$HBF_BASE" "$OUT/configs/hbfine.yaml"
sed 's/^  backend: hbfine$/  backend: flashlab/' "$HBF_BASE" > "$OUT/configs/flashlab.yaml"
cp "$HBM_BASE" "$OUT/configs/fullhbm.yaml"

pids=()
labels=()
started=$SECONDS
for backend in hbfine flashlab fullhbm; do
  for mode in off on; do
    name="${backend}-${mode}"
    dir="$OUT/$name"
    mkdir -p "$dir"
    memory="$OUT/configs/${backend}.yaml"
    cmd=(python3 frontend/tools/run_qwen_simulations.py
      --source "$SOURCE" --params "$PARAMS" --runner "$RUNNER"
      --architecture "$ARCH" --memory "$memory" --out "$dir"
      --flashaccel-config "$POLICY" --prefetch "$mode"
      --max-cycles "$MAX_CYCLES")
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
    printf 'OK\n' > "$OUT/${labels[$i]}/status.txt"
    echo "PROCESS_OK ${labels[$i]}"
  else
    code=$?
    if [[ "$code" == 124 || "$code" == 137 ]]; then
      printf 'TIMEOUT\n' > "$OUT/${labels[$i]}/status.txt"
      echo "PROCESS_TIMEOUT ${labels[$i]} exit=$code" >&2
    else
      printf 'FAILED\n' > "$OUT/${labels[$i]}/status.txt"
      echo "PROCESS_FAILED ${labels[$i]} exit=$code" >&2
    fi
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
      ((.effective_prefetch_mode == "disabled_fullhbm") or
       (.effective_prefetch_mode == "on") or
       (.effective_prefetch_mode == "off"))
    ' "$result" >/dev/null; then
    echo "DATA_OR_CYCLE_FAILED $name" >&2
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

for mode in off on; do
  if [[ -s "$OUT/fullhbm-${mode}/result.json" ]] &&
     ! jq -e '.effective_prefetch_mode == "disabled_fullhbm" and
             (.prefetch_stats.queued // 0) == 0 and
             (.prefetch_stats.admitted // 0) == 0' \
        "$OUT/fullhbm-${mode}/result.json" >/dev/null; then
    echo "FULLHBM_PREFETCH_MODE_MISMATCH $mode" >&2
    failed=1
  fi
done

printf 'case,status,effective_prefetch_mode,cycles,wall_seconds,python_cpu_seconds,cpp_cpu_seconds,python_peak_rss_kib,cpp_peak_rss_kib,peak_rss_kib,queued,admitted,completed,resident_read_bytes,fallback_read_bytes,canceled\n' \
  > "$OUT/summary.csv"
for name in "${labels[@]}"; do
  result="$OUT/$name/result.json"
  if [[ -s "$result" ]]; then
    jq -r --arg name "$name" --arg status "$(cat "$OUT/$name/status.txt" 2>/dev/null || echo OK)" '[ $name, $status, (.effective_prefetch_mode // "unknown"),
      (.timing.total_cycles // 0), (.elapsed_seconds // 0),
      (.python_cpu_seconds // 0), (.cpp_cpu_seconds // 0),
      (.python_peak_rss_kib // 0), (.cpp_peak_rss_kib // 0),
      (.peak_rss_kib // 0),
      (.prefetch_stats.queued // 0), (.prefetch_stats.admitted // 0),
      (.prefetch_stats.completed // 0),
      (.prefetch_stats.resident_read_bytes // 0),
      (.prefetch_stats.fallback_read_bytes // 0),
      (.prefetch_stats.canceled // 0) ] | @csv' "$result" \
      >> "$OUT/summary.csv"
  else
    printf '%s,%s,unknown,0,0,0,0,0,0,0,0,0,0,0,0,0\n' \
      "$name" "$(cat "$OUT/$name/status.txt" 2>/dev/null || echo FAILED)" \
      >> "$OUT/summary.csv"
  fi
done

echo "FINISHED elapsed=${elapsed}s failures=$failed output=$OUT"
exit "$failed"
