#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

JOB_SCRIPT="$ROOT_DIR/src/bench_job_cpu.sh"

check_deps build
echo "[build] Compilation du binaire…"
make -C "$ROOT_DIR/src" PREFIX="$ROOT_DIR" bench
chmod +x "$JOB_SCRIPT"
echo "[build] OK"

# Optionnel: préparer un environnement Conda pour le bench GPU
setup_conda_env() {
	echo "[build] Préparation de l'environnement Conda (optionnel)…"
	# Tenter d'initialiser conda si la fonction/commande n'est pas déjà dispo
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
		echo "[build] Conda introuvable. Étape Conda ignorée."
		return 0
	fi

	# Créer l'env 'bench' si absent
	if conda env list | awk '{print $1}' | grep -qx bench; then
		echo "[build] Env Conda 'bench' déjà présent."
	else
		echo "[build] Création de l'env Conda 'bench'…"
		conda create -y -n bench python=3.10 pip numpy numba pyopencl || {
			echo "[build] Échec création env 'bench' (Conda). Étape Conda ignorée." >&2
			return 0
		}
	fi

	# Afficher le chemin de l'interpréteur et recommandation BENCH_PYTHON
	local py_path
	if py_path=$(conda run -n bench python -c 'import sys; print(sys.executable)' 2>/dev/null); then
		echo "[build] Python de l'env 'bench': $py_path"
		echo "[build] Vous pouvez utiliser cet interpréteur avec Slurm via: export BENCH_PYTHON=\"$py_path\""
	else
		echo "[build] Astuce: définissez BENCH_PYTHON vers l'interpréteur de l'env 'bench' (ex: ~/miniconda3/envs/bench/bin/python)"
	fi

	# Option: installation de bibliothèques supplémentaires GPU (à la charge de l'utilisateur)
	echo "[build] Dépendances installées: numpy, numba, pyopencl."
	echo "[build] Pour PyTorch/CuPy, installez selon votre CUDA (exemples):"
	echo "        conda install -n bench -y -c pytorch pytorch pytorch-cuda=12.1 -c nvidia"
	echo "        pip install cupy-cuda12x  # ou cupy-cuda11x selon votre stack"
}

setup_conda_env
