#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

JOB_SCRIPT="$ROOT_DIR/src/bench_job_gpu.sh"
JOB_NAME="bench_gpu_node"

# Valeurs par défaut (écrasées par arguments CLI)
BENCH_DURATION=3.0
BENCH_REPEATS=5
BENCH_VERBOSE=0
INCLUDE_NODES=""
EXCLUDE_NODES=""
LIMIT_NODES=""
WARMUP_STEPS=5
VRAM_FRAC=0.8
GPU_WALLTIME_FACTOR=10
BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-}"  # on laisse la possibilité d'être pré-positionné

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) BENCH_DURATION="${2:?}"; shift 2 ;;
        --repeats) BENCH_REPEATS="${2:?}"; shift 2 ;;
        --verbose) BENCH_VERBOSE=1; shift ;;
        --include) INCLUDE_NODES="${2:?}"; shift 2 ;;
        --exclude) EXCLUDE_NODES="${2:?}"; shift 2 ;;
        --limit) LIMIT_NODES="${2:?}"; shift 2 ;;
        --warmup) WARMUP_STEPS="${2:?}"; shift 2 ;;
        --vram-frac) VRAM_FRAC="${2:?}"; shift 2 ;;
        --) shift; break ;;
        *) echo "[submit-gpu] option inconnue: $1" >&2; exit 1 ;;
    esac
done

# Activation / vérification conda côté front afin de propager les variables (CONDA_PREFIX, PATH, etc.) au job.
ensure_conda_env() {
    local target_env="$BENCH_CONDA_ENV"
    # Si aucun env explicitement demandé mais un env actif existe, l'utiliser.
    if [[ -z "$target_env" && -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        target_env="$CONDA_DEFAULT_ENV"; BENCH_CONDA_ENV="$target_env"; export BENCH_CONDA_ENV
    fi
    if [[ -z "$target_env" ]]; then
        echo "[submit-gpu] ERREUR: aucun environnement conda actif et BENCH_CONDA_ENV non défini." >&2
        echo "             Activez un env: 'conda activate bench' ou exportez BENCH_CONDA_ENV=bench." >&2
        exit 1
    fi

    # Charger hook conda si nécessaire pour pouvoir activer.
    if ! command -v conda >/dev/null 2>&1; then
        if [[ -n "${CONDA_EXE:-}" ]]; then
            eval "$("$CONDA_EXE" shell.bash hook)" >/dev/null 2>&1 || true
        elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
            # shellcheck source=/dev/null
            source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
        elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
            # shellcheck source=/dev/null
            source "$HOME/anaconda3/etc/profile.d/conda.sh" || true
        fi
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "[submit-gpu] ERREUR: conda introuvable sur le nœud de soumission." >&2
        exit 1
    fi

    # Si l'env actif correspond déjà, ok.
    local cur_env="${CONDA_DEFAULT_ENV:-}" cur_pref="${CONDA_PREFIX:-}"
    if [[ "$cur_env" == "$target_env" || ( -n "$cur_pref" && "$cur_pref" =~ /$target_env$ ) ]]; then
        (( BENCH_VERBOSE == 1 )) && echo "[submit-gpu] Env conda actif: $target_env"
        return 0
    fi

    # Sinon tenter activation.
    if ! conda env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {sub(/\*.*/,"",$0);print $1}' | grep -Fxq "$target_env"; then
        echo "[submit-gpu] ERREUR: environnement conda requis '$target_env' introuvable." >&2
        exit 1
    fi
    conda activate "$target_env" 2>/dev/null || {
        echo "[submit-gpu] ERREUR: échec activation de l'environnement '$target_env'." >&2
        exit 1
    }
    (( BENCH_VERBOSE == 1 )) && echo "[submit-gpu] Environnement conda activé: $target_env"
    BENCH_CONDA_ENV="$target_env"; export BENCH_CONDA_ENV
}

ensure_conda_env

check_deps submit

# build préalable
"$SCRIPT_DIR/build.sh"

# Dresse la liste des nœuds avec GPU et leur quantité (une seule fois)
nodes_with_gpu_counts() {
    for node in $(sinfo -h -N -o "%N"); do
        line=$(scontrol show node -o "$node" 2>/dev/null || true)
        gres=$(sed -n 's/.*Gres=\([^ ]*\).*/\1/p' <<<"$line")
        total=0
        if [[ -n "$gres" && "$gres" != "(null)" ]]; then
            total=$(tr ',' '\n' <<<"$gres" | awk -F: '
                $1=="gpu" { n=$NF; gsub(/[^0-9].*/, "", n); s+=n }
                END{print s+0}
            ')
        fi
        if [[ "$total" -gt 0 ]]; then
            echo "$node $total"
        fi
    done
}

# Construire la table "nœud -> nombre de GPU" une fois
declare -A GPU_COUNT
GPU_NODES=()
while read -r _n _c; do
    GPU_COUNT["$_n"]="$_c"
    GPU_NODES+=("$_n")
done < <(nodes_with_gpu_counts)

if (( BENCH_VERBOSE == 1 )); then
    echo "[submit-gpu] Nœuds avec GPU détectés: ${GPU_NODES[*]:-none}"
    for n in "${GPU_NODES[@]}"; do echo "  - $n: ${GPU_COUNT[$n]}"; done
fi

if [[ ${#GPU_NODES[@]} -eq 0 ]]; then
    echo "Aucun nœud avec GPU détecté." >&2
    exit 1
fi

# include
if [[ -n "$INCLUDE_NODES" ]]; then
    IFS=',' read -r -a inc <<<"$INCLUDE_NODES"
    tmp=()
    for n in "${GPU_NODES[@]}"; do for i in "${inc[@]}"; do [[ "$n" == "$i" ]] && tmp+=("$n"); done; done
    GPU_NODES=("${tmp[@]}")
fi

# exclude
if [[ -n "$EXCLUDE_NODES" ]]; then
    IFS=',' read -r -a exc <<<"$EXCLUDE_NODES"
    tmp=()
    for n in "${GPU_NODES[@]}"; do keep=1; for e in "${exc[@]}"; do [[ "$n" == "$e" ]] && keep=0; done; (( keep )) && tmp+=("$n"); done
    GPU_NODES=("${tmp[@]}")
fi

# limit
if [[ -n "$LIMIT_NODES" ]]; then
    if [[ "$LIMIT_NODES" =~ ^[0-9]+$ ]]; then
        GPU_NODES=("${GPU_NODES[@]:0:LIMIT_NODES}")
    else
        echo "--limit attend un entier." >&2; exit 1
    fi
fi

if [[ ${#GPU_NODES[@]} -eq 0 ]]; then
    echo "[submit-gpu] Aucun nœud GPU à soumettre après filtres." >&2
    exit 0
fi

wall_s=$(( $(estimate_walltime "$BENCH_REPEATS" "$BENCH_DURATION") * GPU_WALLTIME_FACTOR ))
wall=$(fmt_hms "$wall_s")
echo "[submit-gpu] Walltime estimé: $wall (sec=$wall_s)"

for NODE in "${GPU_NODES[@]}"; do
    # Vérifier au dernier moment si le nœud a des GPU occupés
    TOTAL_GPU=${GPU_COUNT[$NODE]:-0}
    echo "[submit-gpu] Soumission sur $NODE (GPU=$TOTAL_GPU)"
    sb_cmd=( sbatch
        --job-name "$JOB_NAME"
        --nodelist "$NODE"
        --nodes 1
        --ntasks-per-node 1
        --cpus-per-task 8
        --gres=gpu:"$TOTAL_GPU"
        --mem=20G
        --time "$wall"
        --output "$OUT_DIR/bench_%N_gpu.out"
        --error "$OUT_DIR/bench_%N_gpu.err"
        --export "ALL,BENCH_ROOT=$ROOT_DIR,BENCH_CONDA_ENV=$BENCH_CONDA_ENV" 
        "$JOB_SCRIPT" --duration "$BENCH_DURATION" --repeats "$BENCH_REPEATS" --warmup "$WARMUP_STEPS" --vram-frac "$VRAM_FRAC" )
    (( BENCH_VERBOSE == 1 )) && sb_cmd+=( --verbose )

    if (( BENCH_VERBOSE == 1 )); then
        printf '[submit-gpu] CMD: '
        printf '%q ' "${sb_cmd[@]}"
        echo
    fi
    "${sb_cmd[@]}"
done

echo "[submit-gpu] Soumissions terminées."
