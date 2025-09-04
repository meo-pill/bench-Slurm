#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

JOB_NAME="bench_cpu_node"

echo "=== Jobs Slurm en cours ($JOB_NAME) ==="
squeue -u "$USER" -n "$JOB_NAME" || true
echo
echo "=== Résultats présents ==="
ls -1 "$RES_DIR"/cpu_*.csv 2>/dev/null | wc -l | xargs -I{} echo "Fichiers résultats: {}"
