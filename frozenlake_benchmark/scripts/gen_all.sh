#!/usr/bin/env bash
set -euo pipefail

python -m frozenlake_benchmark.src.generate_dataset "$@"
