#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

JOB_SCRIPT="$ROOT_DIR/src/bench_job_cpu.sh"
JOB_NAME="bench_cpu_node"

# Lire variables d'environnement par défaut
BENCH_DURATION=${BENCH_DURATION:-2.0}
BENCH_REPEATS=${BENCH_REPEATS:-3}
BENCH_VERBOSE=${BENCH_VERBOSE:-0}
INCLUDE_NODES=${INCLUDE_NODES:-}
EXCLUDE_NODES=${EXCLUDE_NODES:-}
LIMIT_NODES=${LIMIT_NODES:-}
ONLY_NEW=${ONLY_NEW:-0}

check_deps submit

# build préalable
"$SCRIPT_DIR/build.sh"

echo "[submit-cpu] Détection des nœuds pour soumission (idle ∪ avec bench_gpu en cours)..."

# Nœuds idle
mapfile -t IDLE_NODES < <(idle_nodes)

# Nœuds où un job GPU de bench tourne actuellement
get_nodes_running_bench_gpu() {
	local job_name="bench_gpu_node"
	squeue -h --states=RUNNING -o "%N %j" \
		| awk -v jn="$job_name" '$2==jn{print $1}' \
		| while read -r nodelist; do scontrol show hostnames "$nodelist"; done \
		| sort -u
}
mapfile -t GPU_BENCH_NODES < <(get_nodes_running_bench_gpu || true)

# Union des deux listes
mapfile -t NODES < <( { printf '%s\n' "${IDLE_NODES[@]:-}"; printf '%s\n' "${GPU_BENCH_NODES[@]:-}"; } | awk 'NF' | sort -u )

if [[ ${#NODES[@]} -eq 0 ]]; then
	echo "[submit-cpu] Aucun nœud idle ni avec bench_gpu en cours." >&2
	exit 0
fi

(( BENCH_VERBOSE == 1 )) && {
	echo "[submit-cpu] Idle: ${IDLE_NODES[*]:-none}"
	echo "[submit-cpu] GPU-bench: ${GPU_BENCH_NODES[*]:-none}"
}
echo "[submit-cpu] Nœuds ciblés (union): ${NODES[*]}"

# include
if [[ -n "$INCLUDE_NODES" ]]; then
	IFS=',' read -r -a inc <<<"$INCLUDE_NODES"
	tmp=()
	for n in "${NODES[@]}"; do
		for i in "${inc[@]}"; do [[ "$n" == "$i" ]] && tmp+=("$n"); done
	done
	NODES=("${tmp[@]}")
fi

# exclude
if [[ -n "$EXCLUDE_NODES" ]]; then
	IFS=',' read -r -a exc <<<"$EXCLUDE_NODES"
	tmp=()
	for n in "${NODES[@]}"; do
		keep=1; for e in "${exc[@]}"; do [[ "$n" == "$e" ]] && keep=0; done
		(( keep )) && tmp+=("$n")
	done
	NODES=("${tmp[@]}")
fi

# limit
if [[ -n "$LIMIT_NODES" ]]; then
	if [[ "$LIMIT_NODES" =~ ^[0-9]+$ ]]; then
		NODES=("${NODES[@]:0:LIMIT_NODES}")
	else
		echo "--limit attend un entier." >&2; exit 1
	fi
fi

# only new
if (( ONLY_NEW )); then
	tmp=()
	for n in "${NODES[@]}"; do
			f="$RES_DIR/cpu_$n.csv"
			f_legacy="$RES_DIR/$n.csv"
			if [[ ! -f "$f" && ! -f "$f_legacy" ]]; then
			tmp+=("$n")
		else
				# Si le fichier CPU existe, tester. Sinon, tester le legacy.
				target="$f"; [[ -f "$f" ]] || target="$f_legacy"
				if awk -F, 'FNR==1{next} {c++} END{exit !(c==0)}' "$target" ; then
				tmp+=("$n")
			fi
		fi
	done
	NODES=("${tmp[@]}")
fi

if [[ ${#NODES[@]} -eq 0 ]]; then
	echo "[submit-cpu] Aucun nœud à soumettre après filtres." >&2
	exit 0
fi

wall_s=$(estimate_walltime)
wall=$(fmt_hms "$wall_s")
echo "[submit-cpu] Walltime estimé: $wall (sec=$wall_s)"

for NODE in "${NODES[@]}"; do
	# Calculer CPU libres sur le nœud
	line=$(scontrol show node -o "$NODE" 2>/dev/null || true)
	tot=$(sed -n 's/.*CPUTot=\([0-9]*\).*/\1/p' <<<"$line")
	alloc=$(sed -n 's/.*CPUAlloc=\([0-9]*\).*/\1/p' <<<"$line")
	free=$(( tot - alloc ))
	if [[ -z "${free:-}" || "$free" -le 0 ]]; then
		echo "[submit-cpu] $NODE: aucun CPU libre (tot=$tot alloc=$alloc), on saute."
		continue
	fi
	echo "[submit-cpu] Soumission sur $NODE avec $free CPU libres"
	sb_cmd=( sbatch
			--job-name "$JOB_NAME"
			--nodelist "$NODE"
			--nodes 1
			--ntasks-per-node 1
			--cpus-per-task "$free"
			--mem=0
			--time "$wall"
			--output "$OUT_DIR/bench_%N_cpu.out"
			--error "$OUT_DIR/bench_%N_cpu.err"
			--export "ALL,BENCH_ROOT=$ROOT_DIR,BENCH_DURATION=$BENCH_DURATION,BENCH_REPEATS=$BENCH_REPEATS,BENCH_VERBOSE=$BENCH_VERBOSE"
			"$JOB_SCRIPT" )

	if (( BENCH_VERBOSE == 1 )); then
		printf '[submit-cpu] CMD: '
		printf '%q ' "${sb_cmd[@]}"
		echo
	fi
	"${sb_cmd[@]}"
done

echo "[submit-cpu] Soumissions terminées."

