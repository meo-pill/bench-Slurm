# Bench CPU/GPU Slurm (mono & multi)

Outil de benchmark CPU & GPU orchestré par Slurm :

- Compilation d’un binaire CPU OpenMP (portable + binaire natif par hôte si possible)
- Soumission de jobs CPU (non exclusifs, allouant **tous les CPU du nœud** via `CPUTot`)
- Soumission de jobs GPU (tous les GPU du nœud, VRAM adaptative)
- Résultats persistant en CSV (historique cumulatif)
- Classements (« top ») multi‑critères CPU et GPU
- Ajustement dynamique de la taille des buffers GPU selon la VRAM disponible

## Aperçu

- CPU : un job par nœud ciblé (tous les nœuds vus par `sinfo -N` après filtres). Pas de `--exclusive`; on alloue `--cpus-per-task` au nombre de CPU **libres** (CPUTot - CPUAlloc) au moment de la soumission.
- GPU : un job par nœud détecté avec GPUs (via `scontrol show node` / Gres). Alloue tous les GPU (`--gres=gpu:<total>`) et 8 CPU.
- Deux modes par backend : monothread (1 thread / 1 GPU) et multi (tous les threads / tous les GPU disponibles ; fallback mono si un seul GPU).
- Chaque mode est répété N fois → moyenne + écart-type.
- Résultats CSV cumulés (jamais écrasés). Classements CPU/GPU par meilleur run, dernier run, top global ou moyenne historique.

Répertoires:

- `bin/` — binaire `cpu_bench` compilé (OpenMP)
- `results/` — fichiers CSV CPU/GPU (`cpu_<node>.csv`, `gpu_<node>.csv`)
- `outputs/` — logs Slurm (`bench_<node>.out/.err`)

Fichiers principaux / scripts :

- `main.sh` — routeur CLI (build | submit | submit_cpu | submit_gpu | top | status | list)
- `src/cmd/*.sh` — commandes modulaires
  - `build.sh` (compilation + préparation env Conda facultative)
  - `submit.sh` (routeur auto GPU puis CPU) — NOTE : ne supporte pas `--cpu/--gpu` (utiliser `submit_cpu` / `submit_gpu`)
  - `submit_cpu.sh` / `submit_gpu.sh`
  - `top.sh`, `status.sh`, `list.sh`, `cleanup_err_empty.sh`
- `src/bench_job_cpu.sh` — script sbatch CPU (mono + multi)
- `src/bench_job_gpu.sh` — script sbatch GPU (mono + multi pour chaque backend)
- `src/cpu_bench.c` — micro‑benchmark OpenMP (auto‑adapté à `OMP_NUM_THREADS`)
- `src/gpu_bench.py` — orchestration + CSV GPU
- `src/gpu_bench_core.py` — kernels / logique VRAM / multi‑GPU

## Prérequis

- Slurm : `sinfo`, `sbatch`, `squeue`, `scontrol`
- Build : `make`, `gcc` ou `clang` (+ OpenMP), `libm`
- Shell : `awk`, `sort`, `nl`, `tr`
- Python / GPU : environnement Conda **actif** avec Python 3.x et ≥1 backend parmi `torch`, `cupy`, `numba` (OpenCL retiré)

## Compilation

```bash
./main.sh build
```

Produit :

- `bin/cpu_bench` (portable, `-march=x86-64`)
- À l’exécution d’un job CPU, un binaire natif optimisé (`bin/bench-<hostname>`) peut être compilé à la volée (fallback sur le portable si échec).

Conda : si disponible, `build` crée/actualise l’environnement `bench` (packages de base : `python`, `pip`, `numpy`, `numba`) et suggère l’installation de `pytorch` / `cupy` selon votre stack CUDA.

## Utilisation rapide

- Soumettre en mode automatique (tente GPU puis CPU) :

```bash
./main.sh submit
```

- Forcer CPU uniquement :

```bash
./main.sh submit_cpu
```

- Forcer GPU uniquement :

```bash
./main.sh submit_gpu
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
- `submit` — routeur auto : lance `submit_gpu` puis `submit_cpu` (échec global seulement si les deux échouent)
- `submit_cpu` — jobs CPU sur *tous les nœuds visibles* (`sinfo -N`), allocation de tous les CPU déclarés (`CPUTot`) du nœud (non exclusif). Note: si d'autres jobs consomment déjà des cœurs, Slurm peut retarder/ajuster l'allocation.
- `submit_gpu` — jobs GPU sur les nœuds disposant de GPU (alloue tous les GPU du nœud)
- `top` — affiche les classements des nœuds
- `status` — affiche les jobs en cours et une synthèse des résultats
- `list` — liste tous les nœuds du cluster et le nombre de runs enregistrés

Remarques GPU :

- Le runner écrit **une ligne** par exécution dans `results/gpu_<node>.csv` avec de nombreuses métriques (scores + utilisation VRAM par backend).
- Le « top » agrège les backends disponibles en prenant la **moyenne** des colonnes `<backend>_mono_avg` / `<backend>_multi_avg` présentes.

### Note: support OpenCL retiré

Le support OpenCL (pyopencl) a été retiré. Motif: les clusters ciblés exposent déjà CUDA (pilotes + runtime NVIDIA) mais n'intègrent généralement pas les composants supplémentaires nécessaires à OpenCL (ICD loader, bibliothèques fournisseur / paquets spécifiques). Ajouter et maintenir ces couches demanderait une configuration hors standard côté administrateurs sans bénéfice direct pour le bench (les mêmes GPU sont déjà couverts via CUDA avec torch/cupy/numba). Pour réduire la complexité (dépendances, temps d'installation, chemins variables) et éviter des cas de panne silencieuse, OpenCL a donc été désactivé. Les anciennes colonnes OpenCL éventuellement présentes dans des CSV ou notebooks historiques peuvent être ignorées; elles ne seront plus générées.

Conda (obligatoire GPU) :

- Le script Python vérifie qu’un environnement Conda est actif (`CONDA_DEFAULT_ENV` / `CONDA_PREFIX`). Sinon : échec (rc=3).
- Option manuelle : `gpu_bench.py --conda-env <nom>` pour forcer le nom attendu.
- `bench_job_gpu.sh` tente d’activer `BENCH_CONDA_ENV` (défaut `bench`) si `BENCH_PYTHON` n’est pas défini.

## Options communes

- `-r, --repeats N` — nombre de répétitions par mode (défaut: 3)
- `-d, --duration S` — durée d’une répétition en secondes (défaut: 2.0)
- `--include n1,n2` — ne garder que ces nœuds
- `--exclude nX,nY` — exclure ces nœuds
- `--limit N` — limiter au N premiers nœuds après filtres
- `--only-new` — (TODO / non implémenté actuellement dans la logique de filtrage) prévu pour ne lancer que sur les nœuds sans résultats
- `--verbose` — sortie plus détaillée (traces de soumission, commandes sbatch)

Options GPU supplémentaires (passées uniquement via arguments maintenant):

- `--vram-frac F` — fraction de VRAM cible (0.05–0.95, défaut 0.80)
- `--warmup N` — itérations de warmup GPU (0–50, défaut 5) avant chaque mesure

Options de « top » (mutuellement exclusives, défaut `--unique`):

- `--unique` — meilleur run par nœud
- `--unique-last` — dernier run par nœud
- `--top10` — top 10 de toutes les runs (sans agrégation par nœud)
- `--by-node-mean` — moyenne (± écart-type) agrégée par nœud

Exemples :

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

## Variables d’environnement utiles

La configuration se fait désormais via arguments CLI (voir sections ci‑dessus). Les variables ci‑dessous restent optionnelles pour l’infrastructure ou la compatibilité:

- `BENCH_PYTHON` — chemin explicite de l'interpréteur Python (sinon `python3` de l'env actif)
- `BENCH_CONDA_ENV` — nom d'environnement conda attendu côté nœud (validation de cohérence)
- `GPU_WALLTIME_FACTOR` — (optionnel) multiplier le walltime estimé GPU (défaut: 10) si défini avant `submit_gpu`

Les anciennes variables `BENCH_VRAM_FRAC`, `BENCH_WARMUP_STEPS`, `BENCH_DURATION`, `BENCH_REPEATS` ne sont plus lues par les scripts de bench; utilisez les flags CLI.
Les filtres/paramètres sont désormais *exclusivement* véhiculés par arguments (pas d'environnement caché) — sauf `BENCH_CONDA_ENV` si vous devez imposer un nom d'environnement à activer sur les nœuds.

## Walltime automatique

Formule CPU (estimée dans `bench_common.sh`) :

```math
wall_cpu_seconds = max( 2 * repeats * duration * 1.5 + 60 , 60 )
```

Pour les jobs GPU : `wall_gpu = wall_cpu_seconds * GPU_WALLTIME_FACTOR` (défaut ×10) pour couvrir la séquence multi‑backend + multi‑GPU.

Ajustez `--repeats` et `--duration` selon le cluster.

## Format des résultats (CSV)

### CPU

`results/cpu_<node>.csv` :

```text
node,mode,threads,runs,duration_s,avg_events_per_s,stddev_events_per_s,timestamp
```

- `mode` ∈ {mono, multi}
- `threads` = 1 (mono) ou tous les CPU alloués (multi)
- `runs` = nombre de répétitions
- `avg/stddev` = moyenne et écart-type des scores « events per second »
- `timestamp` = horodatage ISO 8601 de la ligne

Le fichier cumule l’historique des runs; rien n’est écrasé.

### GPU

`results/gpu_<node>.csv` — **une ligne par exécution** (schéma extensible) :

```text
node,runs,duration_s,timestamp,
  (torch_mono_avg,torch_mono_std,torch_mono_vram_total_MB,torch_mono_vram_used_MB,torch_mono_vram_used_pct,
   torch_multi_avg,torch_multi_std,torch_multi_gpus,torch_multi_vram_total_MB_sum,torch_multi_vram_used_MB_sum,torch_multi_vram_used_pct,
   cupy_mono_...,cupy_multi_...,
   numba_mono_...,numba_multi_...)
```

Pour chaque backend présent (torch / cupy / numba) :

- `*_mono_avg|std` : score mono‑GPU
- `*_mono_vram_*` : VRAM totale du device, VRAM utilisée par les 4 buffers (MB), pourcentage utilisé
- `*_multi_avg|std` : score multi‑GPU (ou répété mono si un seul GPU)
- `*_multi_gpus` : nombre de GPUs utilisés (>=1)
- `*_multi_vram_*_sum` : sommes agrégées sur l’ensemble des GPUs (si multi) ou mono répété

Les colonnes manquantes (backend absent) restent vides.

## Exemples complets (tous paramètres)

Exemple soumission GPU (VRAM cible 70%) avec filtres et verbosité :

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
          --vram-frac 0.70 \
          --warmup 8 \
          submit_gpu
```

Exemple soumission CPU avec le même jeu d'options :

```bash
./main.sh --repeats 5 \
          --duration 3 \
          --include n1,n2 \
          --exclude n3 \
          --limit 2 \
          --only-new \
          --verbose \
          submit_cpu
```

Exemples « top » :

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

CPU :

- Soumission : `--ntasks-per-node=1`, `--cpus-per-task=<CPUTot>`, `--mem=0`, pas de `--exclusive` (permet coexistence avec d’autres jobs).
- Binaire natif auto (une compilation par hôte) pour exploiter `-march=native` quand disponible.
- Verrou fichier (`results/.lock.<host>`) pour éviter concurrence multi-job sur un même nœud.

GPU :

- Soumission : `--ntasks-per-node=1`, `--cpus-per-task=8`, `--gres=gpu:<total>`, `--mem=20G`.
- Taille des buffers ajustée dynamiquement pour viser la fraction donnée par `--vram-frac` (défaut 0.80) avec réduction si OOM.
- Multi‑GPU : exécution parallèle (threads Python) pour torch/cupy, agrégation séquentielle pour numba si nécessaire.
- Colonnes VRAM : suivi (MB) et pourcentage utilisé.

Général :

- Classements stables (`LC_ALL=C`).
- Nettoyage silencieux des fichiers `.err` vides en arrière‑plan (`cleanup_err_empty.sh`).

## Nettoyage

- Supprimer le binaire: `make clean`
- Repartir de zéro côté résultats: `rm -f results/cpu_*.csv results/gpu_*.csv`

## Dépannage

- Pas de nœuds idle: le cluster est peut-être occupé; réessayez plus tard ou utilisez `--include`.
- OpenMP manquant: installez `gcc` (ou `clang`) avec support OpenMP.
- Permissions Slurm: vérifiez que vous pouvez `sbatch` sur la partition par défaut du cluster.

Conda / GPU :

- Création env : `./main.sh build` (ou manuellement `conda create -n bench python=3.10 numpy numba`)
- Forcer interpréteur : `export BENCH_PYTHON=~/miniconda3/envs/bench/bin/python`
- Installer backends :
  - PyTorch : `conda install -n bench -y -c pytorch pytorch pytorch-cuda=12.1 -c nvidia`
  - CuPy : `pip install cupy-cuda12x` (ou variante correspondant à votre version CUDA)
  - (Numba CUDA déjà présent via `numba`, nécessite drivers NVIDIA compatibles)

VRAM / Warmup : utiliser directement `--vram-frac` et `--warmup` (ex: `--vram-frac 0.70 --warmup 10`).

---
Suggestions ou améliorations bienvenues via issues/PRs.
