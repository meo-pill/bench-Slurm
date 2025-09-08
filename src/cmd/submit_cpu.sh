#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

JOB_SCRIPT="$ROOT_DIR/src/bench_job_cpu.sh"
JOB_NAME="bench_cpu_node"

# Paramètres par défaut (écrasés par arguments)
BENCH_DURATION=3.0
BENCH_REPEATS=5
BENCH_VERBOSE=0
INCLUDE_NODES=""
EXCLUDE_NODES=""
LIMIT_NODES=""
ONLY_NEW=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		--duration) BENCH_DURATION="${2:?}"; shift 2 ;;
		--repeats) BENCH_REPEATS="${2:?}"; shift 2 ;;
		--verbose) BENCH_VERBOSE=1; shift ;;
		--include) INCLUDE_NODES="${2:?}"; shift 2 ;;
		--exclude) EXCLUDE_NODES="${2:?}"; shift 2 ;;
		--limit) LIMIT_NODES="${2:?}"; shift 2 ;;
		--only-new) ONLY_NEW=1; shift ;;
		--) shift; break ;;
		*) echo "[submit-cpu] option inconnue: $1" >&2; exit 1 ;;
	esac
done

check_deps submit

# build préalable
"$SCRIPT_DIR/build.sh"

echo "[submit-cpu] Construction de la liste des nœuds (tous les nœuds visibles dans sinfo)."

# Récupérer tous les nœuds connus du cluster.
mapfile -t NODES < <(sinfo -h -N -o '%N')

if [[ ${#NODES[@]} -eq 0 ]]; then
	echo "[submit-cpu] Aucun nœud détecté via sinfo." >&2
	exit 1
fi

(( BENCH_VERBOSE == 1 )) && echo "[submit-cpu] Total nœuds détectés: ${#NODES[@]} => ${NODES[*]}"
echo "[submit-cpu] Nœuds initiaux: ${#NODES[@]}"

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

wall_s=$(estimate_walltime "$BENCH_REPEATS" "$BENCH_DURATION")
wall=$(fmt_hms "$wall_s")
echo "[submit-cpu] Walltime estimé: $wall (sec=$wall_s)"

for NODE in "${NODES[@]}"; do
	# Calculer CPU libres sur le nœud
	line=$(scontrol show node -o "$NODE" 2>/dev/null || true)
	tot=$(sed -n 's/.*CPUTot=\([0-9]*\).*/\1/p' <<<"$line")
	echo "[submit-cpu] Soumission sur $NODE avec $tot CPU(s) total(s)."
	sb_cmd=( sbatch
			--job-name "$JOB_NAME"
			--nodelist "$NODE"
			--nodes 1
			--ntasks-per-node 1
			--cpus-per-task "$tot"
			--exclusive
			--mem=0
			--time "$wall"
			--output "$OUT_DIR/bench_%N_cpu.out"
			--error "$OUT_DIR/bench_%N_cpu.err"
			--export "ALL,BENCH_ROOT=$ROOT_DIR"
			"$JOB_SCRIPT" --duration "$BENCH_DURATION" --repeats "$BENCH_REPEATS" )
	if (( BENCH_VERBOSE == 1 )); then
		sb_cmd+=( --verbose )
	fi

	if (( BENCH_VERBOSE == 1 )); then
		printf '[submit-cpu] CMD: '
		printf '%q ' "${sb_cmd[@]}"
		echo
	fi
	"${sb_cmd[@]}"
done

echo "[submit-cpu] Soumissions terminées."

