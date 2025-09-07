#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Routeur submit générique → CPU/GPU
MODE=${MODE:-auto}  # auto|cpu|gpu

# Analyser un éventuel premier argument --cpu/--gpu
if [[ ${1:-} == "--cpu" ]]; then
    MODE=cpu; shift
    elif [[ ${1:-} == "--gpu" ]]; then
    MODE=gpu; shift
fi

case "$MODE" in
    cpu)
        bash "$SCRIPT_DIR/submit_cpu.sh" "$@"
        # Lancer le cleanup en arrière-plan
        bash "$SCRIPT_DIR/cleanup_err_empty.sh" >/dev/null 2>&1 &
    ;;
    gpu)
        bash "$SCRIPT_DIR/submit_gpu.sh" "$@"
        # Lancer le cleanup en arrière-plan
        bash "$SCRIPT_DIR/cleanup_err_empty.sh" >/dev/null 2>&1 &
    ;;
    auto)
        # Par défaut: appliquer GPU puis CPU; échouer seulement si les deux échouent
        bash "$SCRIPT_DIR/submit_gpu.sh" "$@"; rc_gpu=$?
        if (( rc_gpu != 0 )); then
            echo "[submit] Soumission GPU non aboutie (rc=$rc_gpu)" >&2
        fi
        bash "$SCRIPT_DIR/submit_cpu.sh" "$@"; rc_cpu=$?
        if (( rc_cpu != 0 )); then
            echo "[submit] Soumission CPU non aboutie (rc=$rc_cpu)" >&2
        fi
        # Lancer le cleanup en arrière-plan
        bash "$SCRIPT_DIR/cleanup_err_empty.sh" >/dev/null 2>&1 &
        if (( rc_gpu != 0 && rc_cpu != 0 )); then
            exit 1
        else
            exit 0
    fi ;;
    *)
    echo "Mode inconnu: $MODE (attendu: auto|cpu|gpu)" >&2; exit 1 ;;
esac
