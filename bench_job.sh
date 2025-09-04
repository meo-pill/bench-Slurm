#!/bin/bash
# Slurm job script: exécute le bench mono et multi sur le nœud local
# Prérequis: export BENCH_ROOT par sbatch (fait par main.sh)

set -euo pipefail

ROOT_DIR=${BENCH_ROOT:?BENCH_ROOT non défini}
BIN_DIR="$ROOT_DIR/bin"
RES_DIR="$ROOT_DIR/results"

HOST=$(hostname -s)
CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)}
DUR=${BENCH_DURATION:-2.0}
REPEATS=${BENCH_REPEATS:-3}

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

# Chemin du binaire
BENCH_BIN="$BIN_DIR/cpu_bench"
if [[ ! -x "$BENCH_BIN" ]]; then
  echo "Binaire $BENCH_BIN introuvable, compilation requise." >&2
  exit 1
fi

# Fichier résultat CSV par nœud
CSV="$RES_DIR/$HOST.csv"
if [[ ! -f "$CSV" ]]; then
  echo "node,mode,threads,runs,duration_s,avg_events_per_s,stddev_events_per_s,timestamp" >"$CSV"
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
    score=$("$BENCH_BIN" --duration "$DUR" | awk '/SCORE/{print $2}')
    echo "[$label] run $i/$REPEATS: $score"
    scores+=("$score")
  done
  read -r avg std <<< "$(printf "%s\n" "${scores[@]}" | calc_stats)"
  ts=$(date -Iseconds)
  echo "$HOST,$label,$mode_threads,$REPEATS,$DUR,$avg,$std,$ts" >>"$CSV"
  echo "$label avg=$avg std=$std"
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
