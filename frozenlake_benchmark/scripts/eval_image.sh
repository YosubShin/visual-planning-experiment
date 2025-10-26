#!/usr/bin/env bash
set -euo pipefail

python -m frozenlake_benchmark.src.run_vlm_eval --variant image "$@"
