#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "$root/../.." && pwd)"
mode="${CROSS_BENCH_MODE:-full}"
rounds="${CROSS_BENCH_ROUNDS:-3}"
raw="$root/results/raw"
rscript="${CROSS_RSCRIPT:-Rscript}"

rm -rf "$raw"
mkdir -p "$raw"

run_julia() {
    local round="$1"
    CROSS_BENCH_MODE="$mode" \
    CROSS_BENCH_OUTPUT="$raw/julia-$round.json" \
      julia --startup-file=no --project="$root/julia" "$root/julia/run.jl"
}

run_r() {
    local round="$1"
    CROSS_BENCH_MODE="$mode" \
    CROSS_BENCH_OUTPUT="$raw/r-$round.json" \
      "$rscript" --vanilla "$root/r/run.R"
}

for round in $(seq 1 "$rounds"); do
    if (( round % 2 == 1 )); then
        run_julia "$round"
        run_r "$round"
    else
        run_r "$round"
        run_julia "$round"
    fi
done

julia --startup-file=no --project="$root/julia" \
  "$root/scripts/compare_results.jl" \
  "$raw" "$root/results/comparison.json" "$root/results/comparison.md"
