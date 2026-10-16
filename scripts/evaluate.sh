#!/usr/bin/env bash
# Regenerates the custom QA benchmark and runs the stage-1-only baseline
# vs. two-stage pipeline comparison, writing outputs/benchmark_report.{json,md}.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m eval.seed_queries
python3 -m eval.run_benchmark
