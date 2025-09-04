# Bench CPU Slurm (mono et multi-thread) par nœud

Outil de bench CPU orchestré par Slurm: compilation d’un binaire OpenMP, soumission d’un job exclusif par nœud idle, stockage des résultats en CSV, et commandes pour classer les nœuds.

## Aperçu

- Un job Slurm par nœud idle, en mode exclusif (utilise tous les CPU du nœud).
- Deux mesures par nœud: monothread et multithread (tous les cœurs alloués).
- Chaque mesure est répétée N fois; on calcule moyenne et écart-type.
- Résultats au format CSV par nœud, et « top » pour classer les machines.

Répertoires:

- `bin/` — binaire `cpu_bench` compilé (OpenMP)
- `results/` — fichiers CSV de résultats (`<node>.csv`)
- `outputs/` — logs Slurm (`bench_<node>.out/.err`)

Fichiers principaux:

- `main.sh` — orchestration (build, submit, top, status, list)
- `bench_job.sh` — job Slurm lancé sur un nœud (exécute mono + multi)
- `cpu_bench.c` — micro-benchmark CPU (OpenMP, calcule un score events/s)

## Prérequis

- Slurm: `sinfo`, `sbatch`, `squeue`
- Build: `make`, `gcc` ou `clang` avec support OpenMP, `libm`
- Outils shell: `awk`, `sort`, `nl`, `tr`

## Compilation

```bash
./main.sh build
```

Le binaire est produit dans `bin/cpu_bench`.

## Utilisation rapide

- Soumettre sur tous les nœuds idle (exclusif), 3 répétitions × 2 s:

```bash
./main.sh submit
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
- `submit` — soumet un job sur chaque nœud idle (applique filtres si fournis)
- `top` — affiche les classements des nœuds
- `status` — affiche les jobs en cours et une synthèse des résultats
- `list` — liste tous les nœuds du cluster et le nombre de runs enregistrés

## Options communes

- `-r, --repeats N` — nombre de répétitions par mode (défaut: 3)
- `-d, --duration S` — durée d’une répétition en secondes (défaut: 2.0)
- `--include n1,n2` — ne garder que ces nœuds
- `--exclude nX,nY` — exclure ces nœuds
- `--limit N` — limiter au N premiers nœuds après filtres
- `--only-new` — ne lancer que sur les nœuds sans résultats

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
./main.sh --only-new submit

# Afficher le top 10 toutes runs confondues
./main.sh --top10 top

# Classement par moyenne sur l’historique
./main.sh --by-node-mean top

# Lister nœuds et nombre de runs enregistrés
./main.sh list
```

## Walltime automatique

`submit` estime automatiquement le walltime et le passe à `sbatch --time`:

```text
wall ≈ 2 modes × repeats × duration × 1.5 + 60 s   (minimum 60 s)
```

Ajustez `--repeats` et `--duration` selon le cluster.

## Format des résultats (CSV)

Chaque nœud a son fichier `results/<node>.csv` avec l’en-tête:

```text
node,mode,threads,runs,duration_s,avg_events_per_s,stddev_events_per_s,timestamp
```

- `mode` ∈ {mono, multi}
- `threads` = 1 (mono) ou tous les CPU alloués (multi)
- `runs` = nombre de répétitions
- `avg/stddev` = moyenne et écart-type des scores « events per second »
- `timestamp` = horodatage ISO 8601 de la ligne

Le fichier cumule l’historique des runs; rien n’est écrasé.

## Détails techniques

- Les jobs sont soumis avec `--exclusive`, `--ntasks-per-node=1`, `--cpus-per-task=<CPU du nœud>`.
- Le binaire `cpu_bench` utilise OpenMP et s’adapte à `OMP_NUM_THREADS`.
- Un verrou léger par nœud empêche l’exécution concurrente sur le même nœud (si FS partagé).
- Tri du « top » stable et locale fixée (LC_ALL=C) pour des classements reproductibles.

## Nettoyage

- Supprimer le binaire: `make clean`
- Repartir de zéro côté résultats: `rm -f results/*.csv`

## Dépannage

- Pas de nœuds idle: le cluster est peut-être occupé; réessayez plus tard ou utilisez `--include`.
- OpenMP manquant: installez `gcc` (ou `clang`) avec support OpenMP.
- Permissions Slurm: vérifiez que vous pouvez `sbatch` sur la partition par défaut du cluster.

---
Suggestions ou améliorations bienvenues via issues/PRs.
