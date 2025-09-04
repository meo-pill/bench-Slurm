#!/bin/bash

# Routeur des commandes bench CPU via sous-scripts dans src/cmd/
# Commandes: build | submit | top | status | list

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
CMD_DIR="$ROOT_DIR/src/cmd"

# Valeurs par défaut (exportées pour les sous-scripts)
BENCH_DURATION=${BENCH_DURATION:-2.0}  # secondes par mesure
BENCH_REPEATS=${BENCH_REPEATS:-3}      # répétitions pour moyenne/écart-type
TOP_MODE=${TOP_MODE:-unique}           # unique | unique-last | top10 | by-node-mean
INCLUDE_NODES=${INCLUDE_NODES:-}       # liste séparée par des virgules
EXCLUDE_NODES=${EXCLUDE_NODES:-}       # liste séparée par des virgules
LIMIT_NODES=${LIMIT_NODES:-}           # limite numérique d'envoi
ONLY_NEW=${ONLY_NEW:-0}                # si 1, ne lance que sur nœuds sans résultats
BENCH_VERBOSE=${BENCH_VERBOSE:-0}      # si 1, verbosité accrue
LC_ALL=C; export LC_ALL

usage() {
    echo "Usage: $0 [--repeats N|-r N] [--duration S|-d S] [--include a,b] [--exclude x,y] [--limit N] [--only-new] [--verbose] [--unique|--unique-last|--top10|--by-node-mean] {build|submit|top|status|list}"
}

# Parse CLI flags et commande
cmd=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        build|submit|top|status|list)
            cmd="$1"; shift ;;
        -r|--repeats)
            BENCH_REPEATS="${2:?valeur manquante pour --repeats}"; shift 2 ;;
        -d|--duration)
            BENCH_DURATION="${2:?valeur manquante pour --duration}"; shift 2 ;;
        --include)
            INCLUDE_NODES="${2:?valeur manquante pour --include}"; shift 2 ;;
        --exclude)
            EXCLUDE_NODES="${2:?valeur manquante pour --exclude}"; shift 2 ;;
        --limit)
            LIMIT_NODES="${2:?valeur manquante pour --limit}"; shift 2 ;;
        --only-new)
            ONLY_NEW=1; shift ;;
        --verbose)
            BENCH_VERBOSE=1; shift ;;
        --unique)
            TOP_MODE="unique"; shift ;;
        --unique-last)
            TOP_MODE="unique-last"; shift ;;
        --top10)
            TOP_MODE="top10"; shift ;;
        --by-node-mean)
            TOP_MODE="by-node-mean"; shift ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "Option inconnue: $1" >&2; usage; exit 1 ;;
    esac
done
[[ -z "${cmd:-}" ]] && cmd="submit"

# Export des variables pour les sous-scripts
export BENCH_DURATION BENCH_REPEATS TOP_MODE INCLUDE_NODES EXCLUDE_NODES LIMIT_NODES ONLY_NEW BENCH_VERBOSE

case "$cmd" in
    build)
        bash "$CMD_DIR/build.sh" ;;
    submit)
        bash "$CMD_DIR/submit.sh" ;;
    top)
        bash "$CMD_DIR/top.sh" ;;
    status)
        bash "$CMD_DIR/status.sh" ;;
    list)
        bash "$CMD_DIR/list.sh" ;;
    *)
        usage; exit 1 ;;
esac

