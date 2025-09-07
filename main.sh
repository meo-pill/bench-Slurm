#!/bin/bash

# Routeur des commandes bench CPU/GPU via sous-scripts dans src/cmd/
# Commandes: build | submit | submit_cpu | submit_gpu | top | status | list

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
        cat <<'EOF'
Bench Slurm CPU/GPU
--------------------
Syntaxe générale:
    ./main.sh <commande> [options]

Commandes disponibles:
    build         Compile le binaire CPU et prépare l'env (optionnel)
    submit        Routeur auto: tente GPU puis CPU
    submit_cpu    Soumet uniquement des jobs CPU
    submit_gpu    Soumet uniquement des jobs GPU
    top           Affiche les classements CPU/GPU
    status        Affiche l'état des jobs et un résumé des résultats
    list          Liste des nœuds et nombre de runs
    help|-h|--help Cette aide

Flags globaux (affectent submit/submit_cpu/submit_gpu):
    -r, --repeats N        Répétitions par mode (défaut: 3)
    -d, --duration S       Durée (s) cible d'une répétition (défaut: 2.0)
    --include n1,n2        Restreindre aux nœuds listés
    --exclude nX,nY        Exclure ces nœuds
    --limit N              Limiter le nombre total de nœuds ciblés
    --only-new             Ne lancer que sur nœuds sans résultats (CSV absent)
    --verbose              Sortie verbeuse (soumissions, détails GPU)

Flags spécifiques GPU (submit / submit_gpu uniquement):
    --vram-frac F          Fraction VRAM cible pour ajuster la taille des buffers (0.05..0.95, défaut 0.80)
    --warmup N             Nombre d'itérations de warmup GPU (0..50, défaut 5) avant mesures

Flags spécifiques top:
    --unique                (défaut) Meilleur run par nœud
    --unique-last           Dernier run par nœud
    --top10                 Top 10 tous runs confondus
    --by-node-mean          Moyenne (± écart-type) agrégée par nœud

Comportement de 'submit':
    1. Tente submit_gpu (ignorer si aucun GPU ou échec bénin)
    2. Puis submit_cpu
    Le retour est en erreur seulement si les deux échouent.

Exemples:
    # Soumettre auto (GPU puis CPU) sur 5 nœuds max avec 5 répétitions de 3s
    ./main.sh --repeats 5 --duration 3 --limit 5 submit

    # Forcer GPU uniquement avec 70% VRAM cible
    ./main.sh --vram-frac 0.70 submit_gpu

    # Classement top10
    ./main.sh --top10 top

EOF
}

# Parse CLI flags et commande
cmd=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        build|submit|submit_cpu|submit_gpu|top|status|list)
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
        --vram-frac)
            BENCH_VRAM_FRAC="${2:?valeur manquante pour --vram-frac}"; shift 2 ;;
        --warmup)
            BENCH_WARMUP_STEPS="${2:?valeur manquante pour --warmup}"; shift 2 ;;
        --unique)
            TOP_MODE="unique"; shift ;;
        --unique-last)
            TOP_MODE="unique-last"; shift ;;
        --top10)
            TOP_MODE="top10"; shift ;;
        --by-node-mean)
            TOP_MODE="by-node-mean"; shift ;;
        -h|--help|help)
            usage; exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "Option inconnue: $1" >&2; usage; exit 1 ;;
    esac
done
[[ -z "${cmd:-}" ]] && cmd="submit"

# Export des variables pour les sous-scripts
export BENCH_DURATION BENCH_REPEATS TOP_MODE INCLUDE_NODES EXCLUDE_NODES LIMIT_NODES ONLY_NEW BENCH_VERBOSE BENCH_VRAM_FRAC BENCH_WARMUP_STEPS

case "$cmd" in
    build)
        bash "$CMD_DIR/build.sh" ;;
    submit)
        bash "$CMD_DIR/submit.sh" ;;
    submit_cpu)
        bash "$CMD_DIR/submit_cpu.sh" ;;
    submit_gpu)
        bash "$CMD_DIR/submit_gpu.sh" ;;
    top)
        bash "$CMD_DIR/top.sh" ;;
    status)
        bash "$CMD_DIR/status.sh" ;;
    list)
        bash "$CMD_DIR/list.sh" ;;
    *)
        usage; exit 1 ;;
esac
exit 0