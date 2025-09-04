#!/bin/bash

# Orchestrateur bench CPU (mono et multi-thread) sur nœuds idle via Slurm
# Modes:
#  - build   : compile le binaire
#  - submit  : soumet un job par nœud idle (exclusif) et collecte les résultats
#  - top     : affiche le top des nœuds (mono et multi)
#  - status  : affiche l'état des jobs et des résultats

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
BIN_DIR="$ROOT_DIR/bin"
RES_DIR="$ROOT_DIR/results"
OUT_DIR="$ROOT_DIR/outputs"
JOB_SCRIPT="$ROOT_DIR/bench_job.sh"
JOB_NAME="bench_cpu_node"
BENCH_DURATION=${BENCH_DURATION:-2.0}  # secondes par mesure
BENCH_REPEATS=${BENCH_REPEATS:-3}      # répétitions pour moyenne/écart-type
TOP_MODE=${TOP_MODE:-unique}           # unique | top10 | by-node-mean
INCLUDE_NODES=""                       # liste séparée par des virgules
EXCLUDE_NODES=""                       # liste séparée par des virgules
LIMIT_NODES=""                         # limite numérique d'envoi
ONLY_NEW=0                              # si 1, ne lance que sur nœuds sans résultats
LC_ALL=C
export LC_ALL

mkdir -p "$BIN_DIR" "$RES_DIR" "$OUT_DIR"

build() {
    check_deps build
    echo "[build] Compilation du binaire…"
    make -C "$ROOT_DIR" bench
    chmod +x "$JOB_SCRIPT"
}

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
        for c in sinfo sbatch; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
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
    awk -v r="$BENCH_REPEATS" -v d="$BENCH_DURATION" 'BEGIN{s=int((2*r*d*1.5)+60); if(s<60)s=60; print s}'
}

submit() {
    check_deps submit
    build
    echo "[submit] Détection des nœuds idle…"
    mapfile -t NODES < <(idle_nodes)
    if [[ ${#NODES[@]} -eq 0 ]]; then
        echo "Aucun nœud idle trouvé." >&2
        return 1
    fi
    echo "[submit] ${#NODES[@]} nœud(s) idle: ${NODES[*]}"

    # Filtres include/exclude
    if [[ -n "${INCLUDE_NODES}" ]]; then
        IFS=',' read -r -a inc <<<"$INCLUDE_NODES"
        tmp=()
        for n in "${NODES[@]}"; do
            for i in "${inc[@]}"; do [[ "$n" == "$i" ]] && tmp+=("$n"); done
        done
        NODES=("${tmp[@]}")
    fi
    if [[ -n "${EXCLUDE_NODES}" ]]; then
        IFS=',' read -r -a exc <<<"$EXCLUDE_NODES"
        tmp=()
        for n in "${NODES[@]}"; do
            keep=1
            for e in "${exc[@]}"; do [[ "$n" == "$e" ]] && keep=0; done
            (( keep )) && tmp+=("$n")
        done
        NODES=("${tmp[@]}")
    fi
    if [[ -n "${LIMIT_NODES}" ]]; then
        if [[ "$LIMIT_NODES" =~ ^[0-9]+$ ]]; then
            NODES=("${NODES[@]:0:LIMIT_NODES}")
        else
            echo "--limit attend un entier." >&2; exit 1
        fi
    fi

    # Garder seulement les nœuds sans résultats si demandé
    if (( ONLY_NEW )); then
        tmp=()
        for n in "${NODES[@]}"; do
            f="$RES_DIR/$n.csv"
            if [[ ! -f "$f" ]]; then
                tmp+=("$n")
            else
                # Pas de lignes de données ?
                if awk -F, 'FNR==1{next} {c++} END{exit !(c==0)}' "$f" ; then
                    tmp+=("$n")
                fi
            fi
        done
        NODES=("${tmp[@]}")
    fi

    if [[ ${#NODES[@]} -eq 0 ]]; then
        echo "[submit] Aucun nœud à soumettre après filtres." >&2
        return 0
    fi
    
    local wall_s; wall_s=$(estimate_walltime)
    local wall; wall=$(fmt_hms "$wall_s")
    echo "[submit] Walltime estimé: $wall (sec=$wall_s)"

    for NODE in "${NODES[@]}"; do
        # Double-vérification à l’instant T (meilleure intention, non garantie contre les races)
        if sinfo -h -N -t idle -o '%N' | grep -qx "$NODE"; then
            echo "[submit] Soumission sur $NODE"
            CPUS_NODE=$(sinfo -h -n "$NODE" -o '%c' | tr -d ' ')
            sbatch \
            --job-name "$JOB_NAME" \
            --nodelist "$NODE" \
            --nodes 1 \
            --ntasks-per-node 1 \
            --cpus-per-task "$CPUS_NODE" \
            --exclusive \
            --mem=0 \
                        --time="$wall" \
            --output "$OUT_DIR/bench_%N.out" \
            --error  "$OUT_DIR/bench_%N.err" \
            --export=ALL,BENCH_ROOT="$ROOT_DIR",BENCH_DURATION="$BENCH_DURATION",BENCH_REPEATS="$BENCH_REPEATS" \
            "$JOB_SCRIPT"
        else
            echo "[submit] $NODE n'est plus idle, on saute."
        fi
    done
    
    echo "[submit] Soumissions terminées. Surveillez: squeue -u $USER -n $JOB_NAME"
}

top() {
    check_deps top
    if ! ls "$RES_DIR"/*.csv >/dev/null 2>&1; then
        echo "Aucun résultat trouvé dans $RES_DIR" >&2
        exit 1
    fi
    
    case "$TOP_MODE" in
        unique)
            echo "=== TOP Monothread (meilleur run par nœud) ==="
            awk -F, 'FNR==1{next} $2=="mono" {k=$1; a=$6; s=$7; if(!(k in max)||a>max[k]){max[k]=a; std[k]=s}} END{for(k in max) printf "%s %.3f ± %.3f\n", k, max[k], std[k]}' "$RES_DIR"/*.csv | sort -s -k2,2nr | nl -w2 -s'. '
            echo
            echo "=== TOP Multithread (meilleur run par nœud) ==="
            awk -F, 'FNR==1{next} $2=="multi" {k=$1; a=$6; s=$7; if(!(k in max)||a>max[k]){max[k]=a; std[k]=s}} END{for(k in max) printf "%s %.3f ± %.3f\n", k, max[k], std[k]}' "$RES_DIR"/*.csv | sort -s -k2,2nr | nl -w2 -s'. '
        ;;
        unique-last)
            echo "=== TOP Monothread (dernier run par nœud) ==="
            for f in "$RES_DIR"/*.csv; do n=$(basename "$f" .csv); awk -F, -v n="$n" 'FNR==1{next} $2=="mono"{a=$6;s=$7} END{if(a!="") printf "%s %.3f ± %.3f\n", n, a, s}' "$f"; done | sort -s -k2,2nr | nl -w2 -s'. '
            echo
            echo "=== TOP Multithread (dernier run par nœud) ==="
            for f in "$RES_DIR"/*.csv; do n=$(basename "$f" .csv); awk -F, -v n="$n" 'FNR==1{next} $2=="multi"{a=$6;s=$7} END{if(a!="") printf "%s %.3f ± %.3f\n", n, a, s}' "$f"; done | sort -s -k2,2nr | nl -w2 -s'. '
        ;;
        top10)
            echo "=== TOP 10 Monothread (toutes runs) ==="
            awk -F, 'FNR==1{next} $2=="mono" {printf "%s %.3f ± %.3f\n", $1, $6, $7}' "$RES_DIR"/*.csv | sort -s -k2,2nr | head -10 | nl -w2 -s'. '
            echo
            echo "=== TOP 10 Multithread (toutes runs) ==="
            awk -F, 'FNR==1{next} $2=="multi" {printf "%s %.3f ± %.3f\n", $1, $6, $7}' "$RES_DIR"/*.csv | sort -s -k2,2nr | head -10 | nl -w2 -s'. '
        ;;
        by-node-mean)
            echo "=== Classement Monothread par moyenne de toutes les runs (par nœud) ==="
            awk -F, 'FNR==1{next} $2=="mono" {k=$1; sum[k]+=$6; ss[k]+=$6*$6; n[k]++} END{for(k in n){m=sum[k]/n[k]; v=(ss[k]/n[k])-m*m; if(v<0)v=0; printf "%s %.3f ± %.3f\n", k, m, sqrt(v)}}' "$RES_DIR"/*.csv | sort -s -k2,2nr | nl -w2 -s'. '
            echo
            echo "=== Classement Multithread par moyenne de toutes les runs (par nœud) ==="
            awk -F, 'FNR==1{next} $2=="multi" {k=$1; sum[k]+=$6; ss[k]+=$6*$6; n[k]++} END{for(k in n){m=sum[k]/n[k]; v=(ss[k]/n[k])-m*m; if(v<0)v=0; printf "%s %.3f ± %.3f\n", k, m, sqrt(v)}}' "$RES_DIR"/*.csv | sort -s -k2,2nr | nl -w2 -s'. '
        ;;
        *)
        echo "TOP_MODE inconnu: $TOP_MODE" >&2; exit 1 ;;
    esac
}

status() {
    echo "=== Jobs Slurm en cours ($JOB_NAME) ==="
    squeue -u "$USER" -n "$JOB_NAME" || true
    echo
    echo "=== Résultats présents ==="
    ls -1 "$RES_DIR"/*.csv 2>/dev/null | wc -l | xargs -I{} echo "Fichiers résultats: {}"
}

list() {
    check_deps list
    echo "=== Nœuds du cluster et nombre de runs enregistrés ==="
    mapfile -t NODES < <(sinfo -h -N -o '%N')
    for NODE in "${NODES[@]}"; do
        f="$RES_DIR/$NODE.csv"
        if [[ -f "$f" ]]; then
            runs=$(awk -F, 'FNR==1{next} $2=="mono"{c++} END{print c+0}' "$f")
        else
            runs=0
        fi
        printf "%s %d\n" "$NODE" "$runs"
    done | sort -s -k1,1
}

usage() {
    echo "Usage: $0 [--repeats N|-r N] [--duration S|-d S] [--include a,b] [--exclude x,y] [--limit N] [--only-new] [--unique|--unique-last|--top10|--by-node-mean] {build|submit|top|status|list}"
}

# Parse CLI flags and command
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

case "$cmd" in
    build) build;;
    submit) submit;;
    top) top;;
    status) status;;
    list) list;;
    *) usage; exit 1;;
esac
