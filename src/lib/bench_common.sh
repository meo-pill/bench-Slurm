#!/bin/bash
# Fonctions et variables communes aux commandes du bench
set -euo pipefail

# Déduit la racine du projet (dossier au-dessus de src/)
# Ce fichier est dans src/lib/, donc on remonte de deux niveaux.
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN_DIR="$ROOT_DIR/bin"
RES_DIR="$ROOT_DIR/results"
OUT_DIR="$ROOT_DIR/outputs"
export ROOT_DIR BIN_DIR RES_DIR OUT_DIR

mkdir -p "$BIN_DIR" "$RES_DIR" "$OUT_DIR"

idle_nodes() {
    sinfo -h -N -t idle -o '%N'
}

check_deps() {
    local ctx=${1:-}
    local missing=()
    # commun
    for c in awk sort nl tr; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
    case "$ctx" in
        build)
            command -v make >/dev/null 2>&1 || missing+=("make")
            if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
                missing+=("gcc/clang")
            fi
        ;;
        submit)
            for c in sinfo squeue scontrol sbatch; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
        ;;
        status)
            command -v squeue >/dev/null 2>&1 || missing+=("squeue")
        ;;
        list)
            command -v sinfo >/dev/null 2>&1 || missing+=("sinfo")
        ;;
    esac
    if (( ${#missing[@]} > 0 )); then
        echo "Dépendances manquantes: ${missing[*]}" >&2
        exit 1
    fi
}

# Convertit secondes entières -> HH:MM:SS
fmt_hms() {
    local s=$1
    printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# Estimation walltime: 2 modes (mono+multi) * repeats * duration * 1.5 + 60s marge
estimate_walltime() {
    # Usage: estimate_walltime <repeats> <duration>
    local repeats=${1:-3}
    local duration=${2:-2.0}
    awk -v r="$repeats" -v d="$duration" 'BEGIN{s=int((2*r*d*1.5)+60); if(s<60)s=60; print s}'
}
