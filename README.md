# Bench CPU/GPU Slurm (mono et multi-thread)

Outil de bench CPU/GPU orchestré par Slurm: compilation d’un binaire CPU OpenMP, soumission de jobs exclusifs par nœud, stockage des résultats en CSV, et commandes pour classer les nœuds.

## Aperçu

- Un job Slurm par nœud idle, en mode exclusif (utilise tous les CPU du nœud).
- Deux mesures par nœud: monothread et multithread (tous les cœurs alloués).
- Chaque mesure est répétée N fois; on calcule moyenne et écart-type.
- Résultats au format CSV par nœud, et « top » pour classer les machines.

Répertoires:

- `bin/` — binaire `cpu_bench` compilé (OpenMP)
- `results/` — fichiers CSV CPU/GPU (`cpu_<node>.csv`, `gpu_<node>.csv`)
- `outputs/` — logs Slurm (`bench_<node>.out/.err`)

Fichiers principaux:

- `main.sh` — routeur CLI (build, submit, submit_cpu, submit_gpu, top, status, list)
- `src/cmd/*.sh` — sous-scripts appelés par `main.sh` (build/submit/top/status/list, et variantes CPU/GPU)
- `src/bench_job_cpu.sh` — job Slurm CPU (exécute mono + multi)
- `src/bench_job_gpu.sh` — job Slurm GPU (déclenche le bench GPU Python)
- `src/cpu_bench.c` — micro-benchmark CPU (OpenMP)
- `src/gpu_bench.py` — runner GPU (orchestration/CSV)
- `src/gpu_bench_core.py` — fonctions cœur des benchmarks GPU

## Prérequis

- Slurm: `sinfo`, `sbatch`, `squeue`
- Build: `make`, `gcc` ou `clang` avec support OpenMP, `libm`
- Outils shell: `awk`, `sort`, `nl`, `tr`
- GPU (obligatoire): un environnement Conda actif (voir ci-dessous) contenant Python 3.x et au moins un backend parmi `torch`, `cupy`, `numba`, `pyopencl`.

## Compilation

```bash
./main.sh build
```

Le binaire est produit dans `bin/cpu_bench`.

Si Conda est disponible, `build` prépare aussi un environnement `bench` et y installe des dépendances de base (numpy, numba, pyopencl). Cet environnement (ou un autre équivalent) est requis pour le bench GPU. Activez-le avant exécution côté nœud ou laissez le job GPU l’activer.

## Utilisation rapide

- Soumettre en mode automatique (essaie GPU puis CPU), 3 répétitions × 2 s:

```bash
./main.sh submit
```

- Forcer CPU uniquement:

```bash
./main.sh submit_cpu
# ou
./main.sh submit --cpu
```

- Forcer GPU uniquement:

```bash
./main.sh submit_gpu
# ou
./main.sh submit --gpu
```

- Voir l’état des jobs et le nombre de fichiers résultats:

```bash
./main.sh status
```

- Afficher le classement (par défaut: meilleur run par nœud, moyenne ± écart-type):

```bash
./main.sh top
```

## Commandes

- `build` — compile le binaire
- `submit` — routeur auto: lance d'abord `submit_gpu`, puis `submit_cpu` (échoue seulement si les deux échouent)
- `submit_cpu` — soumet des jobs CPU sur les nœuds idle
- `submit_gpu` — soumet des jobs GPU sur les nœuds avec GPU libres
- `top` — affiche les classements des nœuds
- `status` — affiche les jobs en cours et une synthèse des résultats
- `list` — liste tous les nœuds du cluster et le nombre de runs enregistrés

Remarques GPU:

- Le runner GPU écrit une seule ligne par exécution dans `results/gpu_<node>.csv`.
- Le « top » inclut des sections GPU qui agrègent par moyenne des backends disponibles (torch/cupy/numba/opencl), en mono et multi.

Conda (obligatoire pour GPU):

- Un environnement Conda actif est requis lors de l’exécution du runner GPU.
- Si aucun environnement n’est actif, l’exécution échoue immédiatement (code de retour 3).
- Le runner accepte l’option `--conda-env <name>` pour vérifier que le nom de l’environnement actif correspond.
- Même si vous fournissez `BENCH_PYTHON`, un environnement Conda actif reste nécessaire.

## Options communes

- `-r, --repeats N` — nombre de répétitions par mode (défaut: 3)
- `-d, --duration S` — durée d’une répétition en secondes (défaut: 2.0)
- `--include n1,n2` — ne garder que ces nœuds
- `--exclude nX,nY` — exclure ces nœuds
- `--limit N` — limiter au N premiers nœuds après filtres
- `--only-new` — ne lancer que sur les nœuds sans résultats
- `--verbose` — sortie plus détaillée (traces de soumission, commandes sbatch)

Options de « top » (sélection d’un mode; défaut: `--unique`):

- `--unique` — meilleur run par nœud
- `--unique-last` — dernier run par nœud
- `--top10` — top 10 de toutes les runs (sans agrégation par nœud)
- `--by-node-mean` — moyenne (± écart-type) agrégée par nœud

Exemples:

```bash
# Soumettre avec 5 répétitions de 3 s, seulement sur n1 et n2
./main.sh --repeats 5 --duration 3 --include n1,n2 submit

# Soumettre uniquement sur nœuds sans résultats
./main.sh --only-new submit_cpu

# Afficher le top 10 toutes runs confondues
./main.sh --top10 top

# Classement par moyenne sur l’historique
./main.sh --by-node-mean top

# Lister nœuds et nombre de runs enregistrés
./main.sh list
```

## Sous-scripts et structure

Les commandes sont implémentées comme sous-scripts sous `src/cmd/` et partagent des utilitaires communs dans `src/lib/bench_common.sh`.

- `src/cmd/build.sh` — build CPU et préparation du job script
- `src/cmd/submit.sh` — routeur (auto|cpu|gpu)
- `src/cmd/submit_cpu.sh` — soumission CPU
- `src/cmd/submit_gpu.sh` — soumission GPU (détecte nœuds avec GPU libres)
- `src/cmd/top.sh`, `src/cmd/status.sh`, `src/cmd/list.sh`

Variables d’environnement propagées par `main.sh` aux sous-scripts:

- `BENCH_DURATION`, `BENCH_REPEATS`, `BENCH_VERBOSE`
- `INCLUDE_NODES`, `EXCLUDE_NODES`, `LIMIT_NODES`, `ONLY_NEW`
- `TOP_MODE`
- GPU spécifiques:
  - `BENCH_PYTHON` — chemin de l’interpréteur Python (souvent celui de votre env Conda). Note: un env Conda actif est requis de toute façon.
  - `BENCH_CONDA_ENV` — nom de l’environnement Conda à activer côté nœud (défaut: `bench`). En cas d’échec d’activation, le job GPU échoue.

## Walltime automatique

`submit` estime automatiquement le walltime et le passe à `sbatch --time`:

```text
wall ≈ 2 modes × repeats × duration × 1.5 + 60 s   (minimum 60 s)
```

Ajustez `--repeats` et `--duration` selon le cluster.

## Format des résultats (CSV)

Chaque nœud a son fichier CPU `results/cpu_<node>.csv` avec l’en-tête:

```text
node,mode,threads,runs,duration_s,avg_events_per_s,stddev_events_per_s,timestamp
```

- `mode` ∈ {mono, multi}
- `threads` = 1 (mono) ou tous les CPU alloués (multi)
- `runs` = nombre de répétitions
- `avg/stddev` = moyenne et écart-type des scores « events per second »
- `timestamp` = horodatage ISO 8601 de la ligne

Le fichier cumule l’historique des runs; rien n’est écrasé.

Résultats GPU (`results/gpu_<node>.csv`) — une ligne par exécution avec l’en-tête:

```text
node,runs,duration_s,timestamp,(<backend>_mono_avg,<backend>_mono_std,<backend>_multi_avg,<backend>_multi_std,<backend>_multi_gpus)*
```

où `<backend>` ∈ {torch, cupy, numba, opencl} selon disponibilité.

## Exemples complets (tous paramètres)

Exemple soumission GPU (non exclusive) avec toutes les options CLI et variables d'environnement utiles:

```bash
# (optionnel) forcer l'interpréteur Python et/ou activer un env Conda côté nœud
export BENCH_PYTHON=/chemin/vers/conda/envs/bench/bin/python
export BENCH_CONDA_ENV=bench

# soumettre uniquement sur certains nœuds, en exclure d'autres, limiter le nombre,
# avec 5 répétitions de 3 secondes, uniquement si pas de résultats, en mode verbeux
./main.sh --repeats 5 \
          --duration 3 \
          --include n1,n2 \
          --exclude n3 \
          --limit 2 \
          --only-new \
          --verbose \
          submit --gpu
```

Exemple soumission CPU avec le même jeu d'options:

```bash
./main.sh --repeats 5 \
          --duration 3 \
          --include n1,n2 \
          --exclude n3 \
          --limit 2 \
          --only-new \
          --verbose \
          submit --cpu
```

Exemples « top » avec les différents modes disponibles:

```bash
# meilleur run par nœud (défaut)
./main.sh --unique top

# dernier run par nœud
./main.sh --unique-last top

# top 10 toutes runs confondues
./main.sh --top10 top

# classement par moyenne (± écart-type) par nœud
./main.sh --by-node-mean top
```

## Détails techniques

- Jobs CPU: `--exclusive`, `--ntasks-per-node=1`, `--cpus-per-task=<CPU du nœud>`.
- Jobs GPU: `--ntasks-per-node=1`, `--cpus-per-task=8`, `--mem=20G`, `--gres=gpu:<tous_les_GPU_du_nœud>` (détection automatique du total via Slurm; pas d'`--exclusive`).
- Le binaire `cpu_bench` utilise OpenMP et s’adapte à `OMP_NUM_THREADS`.
- Un verrou léger par nœud empêche l’exécution concurrente sur le même nœud (si FS partagé).
- Tri du « top » stable et locale fixée (LC_ALL=C) pour des classements reproductibles.

Remarques GPU:

- Le bench GPU Python détecte dynamiquement les backends disponibles (torch, cupy, numba, pyopencl).
- Structure séparée: `gpu_bench_core.py` (fonctions de bench) et `gpu_bench.py` (runner + CSV).
- Le job GPU tente d’activer l’environnement Conda `BENCH_CONDA_ENV` (défaut `bench`). Si l’environnement n’est pas actif après cette étape, l’exécution échoue (pas de repli sur `python3`).
- Le runner `gpu_bench.py` peut valider le nom via `--conda-env <name>` et refusera de s’exécuter sans env Conda actif.

## Nettoyage

- Supprimer le binaire: `make clean`
- Repartir de zéro côté résultats: `rm -f results/cpu_*.csv results/gpu_*.csv`

## Dépannage

- Pas de nœuds idle: le cluster est peut-être occupé; réessayez plus tard ou utilisez `--include`.
- OpenMP manquant: installez `gcc` (ou `clang`) avec support OpenMP.
- Permissions Slurm: vérifiez que vous pouvez `sbatch` sur la partition par défaut du cluster.

Conda/pyGPU:

- Créez/activez l’env: `./main.sh build` prépare un env `bench` si Conda est présent.
- Utiliser un Python précis avec Slurm: exportez `BENCH_PYTHON=/chemin/vers/conda/envs/bench/bin/python` avant `submit_gpu`.
- Installation backends (exemples):
  - PyTorch CUDA (ex.): `conda install -n bench -y -c pytorch pytorch pytorch-cuda=12.1 -c nvidia`
  - CuPy (ex.): `pip install cupy-cuda12x` (ajustez selon votre CUDA)

---
Suggestions ou améliorations bienvenues via issues/PRs.
