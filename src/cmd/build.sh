#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

JOB_SCRIPT="$ROOT_DIR/src/bench_job_cpu.sh"

check_deps build
echo "[build] Compilation du binaire…"
make -C "$ROOT_DIR/src" PREFIX="$ROOT_DIR" bench
chmod +x "$JOB_SCRIPT"
echo "[build] OK"
