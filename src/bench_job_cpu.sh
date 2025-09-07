#!/bin/bash
# Slurm job script: exécute le bench mono et multi sur le nœud local
# Prérequis: export BENCH_ROOT par sbatch (fait par main.sh)

set -euo pipefail

ROOT_DIR=${BENCH_ROOT:?BENCH_ROOT non défini}
BIN_DIR="$ROOT_DIR/bin"
RES_DIR="$ROOT_DIR/results"

HOST=$(hostname -s)
CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)}
DUR=3.0
REPEATS=5
VERBOSE=0

# Parsing des arguments transmis par submit_cpu.sh
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)
            DUR="${2:?valeur manquante pour --duration}"; shift 2 ;;
        --repeats)
            REPEATS="${2:?valeur manquante pour --repeats}"; shift 2 ;;
        --verbose)
            VERBOSE=1; shift ;;
        --)
            shift; break ;;
        *)
            echo "[bench_job_cpu] option inconnue: $1" >&2; exit 1 ;;
    esac
done

# Empêcher une exécution concurrente si le répertoire est partagé
lockfile="$RES_DIR/.lock.$HOST"
if ! ( set -o noclobber; : >"$lockfile" ) 2>/dev/null; then
    echo "Un bench est déjà en cours pour $HOST, on quitte." >&2
    exit 0
fi
trap 'rm -f "$lockfile"' EXIT

# Variables pour libs BLAS/OpenMP
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Prépare les binaires dans le dossier bin du projet
GENERIC_BIN="$BIN_DIR/cpu_bench"
NATIVE_BIN="$BIN_DIR/bench-$HOST"

# Construire le binaire natif via le Makefile (dans bin/)
if [[ ! -x "$NATIVE_BIN" ]]; then
    (( VERBOSE == 1 )) && echo "[bench] build native for host=$HOST into $NATIVE_BIN"
    make -C "$ROOT_DIR/src" native-host HOSTNAME="$HOST" PREFIX="$ROOT_DIR" >/dev/null 2>&1 || true
fi

# Choix du binaire: natif si disponible, sinon générique
if [[ -x "$NATIVE_BIN" ]]; then
    BENCH_BIN="$NATIVE_BIN"
    (( VERBOSE == 1 )) && echo "[bench] Using native binary: $BENCH_BIN"
else
    BENCH_BIN="$GENERIC_BIN"
fi
if [[ ! -x "$BENCH_BIN" ]]; then
    echo "Aucun binaire exécutable trouvé (ni natif: $NATIVE_BIN, ni générique: $GENERIC_BIN)." >&2
    exit 1
fi

# Fichier résultat CSV par nœud (préfixé)
CSV="$RES_DIR/cpu_$HOST.csv"
new_header="node,mode,threads,runs,duration_s,avg_events_per_s,stddev_events_per_s,min_events_per_s,max_events_per_s,timestamp"
if [[ -f "$CSV" ]]; then
    # Vérifier si l'en-tête correspond au nouveau schéma, sinon sauvegarder l'ancien
    read -r first_line <"$CSV" || true
    if [[ "$first_line" != "$new_header" ]]; then
        mv "$CSV" "$CSV.bak.$(date +%s)" 2>/dev/null || true
    fi
fi
if [[ ! -f "$CSV" ]]; then
    echo "$new_header" >"$CSV"
fi

calc_stats() {
    # lit des nombres (un par ligne) sur stdin et imprime: avg stddev
    awk '{s+=$1;n++;ss+=$1*$1} END{if(n){m=s/n; v=(ss/n)-(m*m); if(v<0) v=0; printf "%.3f %.3f\n", m, sqrt(v)} else {print "0 0"}}'
}

run_mode() {
    local mode_threads=$1  # 1 ou $CPUS
    local label=$2         # mono|multi
    local i score
    # configure OpenMP
    export OMP_NUM_THREADS=$mode_threads
    # collecte des scores
    scores=()
    for ((i=1;i<=REPEATS;i++)); do
        extra=""
        (( VERBOSE == 1 && i == 1 )) && extra="--verbose"
        # Exécuter en capturant stdout tout en laissant stderr aller au fichier .err de Slurm
        set +e
        output=$("$BENCH_BIN" --duration "$DUR" $extra 2> >(tee >&2))
        rc=$?
        set -e
        if (( rc != 0 )); then
        echo "[$label] run $i/$REPEATS: échec (rc=$rc)" >&2
        continue
        fi
        score=$(awk '/SCORE/{print $2}' <<<"$output")
        if [[ -z "$score" ]]; then
        echo "[$label] run $i/$REPEATS: aucun SCORE détecté" >&2
        continue
        fi
        echo "[$label] run $i/$REPEATS: $score"
        scores+=("$score")
    done
    read -r avg std <<< "$(printf "%s\n" "${scores[@]}" | calc_stats)"
    # Calcul min / max (si scores non vides)
    if ((${#scores[@]})); then
        min_v=${scores[0]}
        max_v=${scores[0]}
        for v in "${scores[@]}"; do
        (( $(awk -v a="$v" -v b="$min_v" 'BEGIN{print (a<b)}') )) && min_v=$v || true
        (( $(awk -v a="$v" -v b="$max_v" 'BEGIN{print (a>b)}') )) && max_v=$v || true
        done
    else
        min_v=0; max_v=0
    fi
    ts=$(date -Iseconds)
    echo "$HOST,$label,$mode_threads,$REPEATS,$DUR,$avg,$std,$min_v,$max_v,$ts" >>"$CSV"
    echo "$label avg=$avg std=$std min=$min_v max=$max_v"
}

# Monothread
run_mode 1 mono

# Multithread (tous les CPU du nœud alloués)
run_mode "$CPUS" multi

# Affichage de synthèse pour les logs Slurm
printf "Host=%s mono(avg)=%.3f multi(avg)=%.3f (threads=%d runs=%d)\n" "$HOST" \
    "$(awk -F, -v h="$HOST" '$1==h && $2=="mono" {print $6; exit}' "$CSV")" \
    "$(awk -F, -v h="$HOST" '$1==h && $2=="multi" {print $6; exit}' "$CSV")" \
    "$CPUS" "$REPEATS"
