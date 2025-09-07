#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

PID_FILE="$OUT_DIR/.cleanup_err_empty.pid"
JOBN_CPU="bench_cpu_node"
JOBN_GPU="bench_gpu_node"

# Éviter les doublons via pidfile
if [[ -f "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE" || true)
    if [[ -n "${old_pid:-}" ]] && ps -p "$old_pid" -o comm= >/dev/null 2>&1; then
        exit 0
    fi
fi
echo $$ > "$PID_FILE"

# Attendre la fin de tous les jobs bench (CPU/GPU) du user
while true; do
    # squeue retourne 0 même s'il n'y a rien; on teste le nombre de lignes
    cnt=$(squeue -h -u "$USER" -n "$JOBN_CPU,$JOBN_GPU" | wc -l | tr -d ' ')
    if [[ "$cnt" == "0" ]]; then
        break
    fi
    sleep 15
done

# petite marge pour que Slurm flush les fichiers
sleep 5

# Supprimer les .err vides (cpu/gpu)
find "$OUT_DIR" -maxdepth 1 -type f \( -name 'bench_*_cpu.err' -o -name 'bench_*_gpu.err' \) -size 0 -print -delete || true

# Nettoyage pidfile
rm -f "$PID_FILE" || true

exit 0
