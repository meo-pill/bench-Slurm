"""Runner d'orchestration du bench GPU.

Ce script gère:
- le parsing des arguments,
- l'exécution mono et multi pour chaque backend disponible,
- le calcul moyenne/écart-type sur N répétitions,
- et l'écriture d'une ligne consolidée dans outputs/gpu_<node>.csv.

Les fonctions de bench et de listing des devices sont importées depuis
gpu_bench_core.py afin de séparer la logique cœur et l'orchestration.
"""
import argparse
import sys
import os
import socket
from datetime import datetime
import math

from gpu_bench_core import (
    bench_torch, bench_cupy, bench_numba,
    list_devices_torch, list_devices_cupy, list_devices_numba,
    bench_torch_multi, bench_cupy_multi, set_vram_target, set_warmup_steps,
)


def display_result(backend: str, mode: str, threads: int, duration: float, avg: float, std: float, runs: int) -> None:
    """Affichage standardisé d'un résultat de bench.

    Imprime des lignes faciles à parser:
    BACKEND <backend>\n
    MODE <mode>\n
    THREADS <threads>\n
    DURATION <duration>\n
    SCORE <avg>\n
    STD <std>\n
    RUNS <runs>
    """
    print(f'BACKEND {backend}')
    print(f'MODE {mode}')
    print(f'THREADS {threads}')
    print(f'DURATION {duration:.3f}')
    print(f'SCORE {avg:.3f}')
    print(f'STD {std:.3f}')
    print(f'RUNS {runs}')

def ensure_conda_active(expected_name: str | None = None) -> None:
    """Vérifie qu'un environnement conda est actif, sinon bloque l'exécution.

    Si expected_name est fourni, vérifie que l'env actif correspond.
    En cas d'échec, imprime un message d'erreur sur stderr et quitte avec code 3.
    """
    env = os.environ.get('CONDA_DEFAULT_ENV')
    prefix = os.environ.get('CONDA_PREFIX')
    if not env and not prefix:
        print("[error] Un environnement conda actif est requis. Activez-le (conda activate <env>) avant d'exécuter gpu_bench.py.", file=sys.stderr)
        sys.exit(3)
    if expected_name:
        ok = False
        if env and env == expected_name:
            ok = True
        if prefix and (prefix.rstrip('/').endswith('/'+expected_name) or os.path.basename(prefix.rstrip('/')) == expected_name):
            ok = True
        if not ok:
            cur = env or prefix or '(inconnu)'
            print(f"[error] L'environnement conda actif ('{cur}') ne correspond pas à l'environnement requis '{expected_name}'.", file=sys.stderr)
            sys.exit(3)


def main():
    p = argparse.ArgumentParser(description='GPU compute benchmark (Python). Exécute toujours mono et multi pour chaque backend disponible.')
    p.add_argument('--duration', type=float, default=2.0, help='durée cible en secondes')
    p.add_argument('--size', type=int, default=1<<23, help='taille du vecteur (peut être réduit si OOM)')
    p.add_argument('--repeats', type=int, default=3, help='nombre de répétitions pour moyenne/écart-type')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--conda-env', type=str, default=None, help="nom de l'environnement conda requis (obligatoire: un conda actif doit être présent)")
    # outputs sous la racine du projet (parent de src)
    p.add_argument('--csv-dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs'), help='répertoire pour stocker le CSV consolidé')
    p.add_argument('--node', type=str, default=socket.gethostname().split('.')[0], help='nom du nœud pour les CSV')
    p.add_argument('--vram-frac', type=float, default=None, help='fraction VRAM cible (0.05-0.95) pour ajuster la taille des buffers GPU')
    p.add_argument('--warmup', type=int, default=None, help='override du nombre d\'itérations de warmup (0..50)')
    args = p.parse_args()

    # Vérifie conda actif et (optionnellement) le nom d'env requis
    ensure_conda_active(args.conda_env)

    backends = ['torch','cupy','numba']

    # Override VRAM target if requested
    if args.vram_frac is not None:
        set_vram_target(args.vram_frac)
        if args.verbose:
            print(f"[cfg] VRAM target fraction override: {args.vram_frac:.3f}")
    if args.warmup is not None:
        set_warmup_steps(args.warmup)
        if args.verbose:
            print(f"[cfg] Warmup steps override: {args.warmup}")

    # Prépare CSV dir
    csv_dir = args.csv_dir
    os.makedirs(csv_dir, exist_ok=True)

    def write_csv_row(results):
        """Écrit une seule ligne récapitulative dans gpu_<node>.csv.

        Nouveau schéma colonnes par backend (mono puis multi):
          <be>_mono_avg,<be>_mono_std,<be>_mono_vram_total_MB,<be>_mono_vram_used_MB,<be>_mono_vram_used_pct,
          <be>_multi_avg,<be>_multi_std,<be>_multi_gpus,<be>_multi_vram_total_MB_sum,<be>_multi_vram_used_MB_sum,<be>_multi_vram_used_pct

        Si un ancien fichier existe (en-tête différent) il est renommé *.bak puis recréé.
        """
        path = os.path.join(csv_dir, f"gpu_{args.node}.csv")
        be_cols = []
        for be in backends:
            be_cols += [
                f"{be}_mono_avg", f"{be}_mono_std", f"{be}_mono_vram_total_MB", f"{be}_mono_vram_used_MB", f"{be}_mono_vram_used_pct",
                f"{be}_multi_avg", f"{be}_multi_std", f"{be}_multi_gpus", f"{be}_multi_vram_total_MB_sum", f"{be}_multi_vram_used_MB_sum", f"{be}_multi_vram_used_pct"
            ]
        header = 'node,runs,duration_s,timestamp,' + ','.join(be_cols) + '\n'
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    first = f.readline()
                if first != header:
                    os.replace(path, path + '.bak')
            except Exception:
                pass
        if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            with open(path, 'w') as f:
                f.write(header)
        ts = datetime.now().isoformat(timespec='seconds')
        row_vals = [
            args.node,
            str(results.get('runs', args.repeats)),
            f"{results.get('duration_s', args.duration):.3f}",
            ts,
        ]
        for col in be_cols:
            v = results.get(col, '')
            if isinstance(v, float):
                row_vals.append(f"{v:.3f}")
            else:
                row_vals.append(str(v))
        with open(path, 'a') as f:
            f.write(','.join(row_vals) + '\n')

    def calc_stats(vals):
        n = len(vals)
        if n == 0:
            return 0.0, 0.0
        s = sum(vals)
        ss = sum(v*v for v in vals)
        m = s / n
        v = (ss / n) - (m * m)
        if v < 0:
            v = 0.0
        return m, math.sqrt(v)

    last_err = None
    any_ok = False
    aggregate = { 'runs': args.repeats, 'duration_s': args.duration }
    for be in backends:
        try:
            printed = 0
            if be == 'torch':
                # Mono-GPU
                vals = []
                for i in range(args.repeats):
                    s1 = bench_torch(args.duration, 0, args.size, args.verbose)
                    vals.append(s1)
                    if args.verbose:
                        print(f"[torch mono] run {i+1}/{args.repeats}: {s1:.3f}")
                avg, std = calc_stats(vals)
                display_result(be, 'mono', 1, args.duration, avg, std, len(vals))
                aggregate['torch_mono_avg'] = avg
                aggregate['torch_mono_std'] = std
                # VRAM mono torch
                vinfo = getattr(bench_torch, 'last_vram', None)
                if vinfo:
                    aggregate['torch_mono_vram_total_MB'] = vinfo['total_bytes']/1e6
                    aggregate['torch_mono_vram_used_MB'] = vinfo['used_bytes']/1e6
                    aggregate['torch_mono_vram_used_pct'] = (vinfo['used_bytes']/vinfo['total_bytes']*100.0) if vinfo['total_bytes'] else 0.0
                printed += 1
                any_ok = True
                # Multi-GPU
                devs = list_devices_torch()
                threads_count = len(devs) if len(devs) > 1 else 1
                vals = []
                for i in range(args.repeats):
                    if len(devs) > 1:
                        s = bench_torch_multi(args.duration, devs, args.size, args.verbose)
                    else:
                        s = bench_torch(args.duration, 0, args.size, args.verbose)
                    vals.append(s)
                    if args.verbose:
                        print(f"[torch multi] run {i+1}/{args.repeats}: {s:.3f}")
                avg, std = calc_stats(vals)
                display_result(be, 'multi', threads_count, args.duration, avg, std, len(vals))
                aggregate['torch_multi_avg'] = avg
                aggregate['torch_multi_std'] = std
                aggregate['torch_multi_gpus'] = threads_count
                # VRAM multi torch
                if threads_count > 1:
                    vinfo_m = getattr(bench_torch_multi, 'last_vram', None)
                    if vinfo_m:
                        total_sum = vinfo_m.get('total_bytes_sum') or 0
                        used_sum = vinfo_m.get('used_bytes_sum') or 0
                        aggregate['torch_multi_vram_total_MB_sum'] = total_sum/1e6
                        aggregate['torch_multi_vram_used_MB_sum'] = used_sum/1e6
                        aggregate['torch_multi_vram_used_pct'] = (used_sum/total_sum*100.0) if total_sum else 0.0
                else:
                    # fallback répète mono
                    vinfo = getattr(bench_torch, 'last_vram', None)
                    if vinfo:
                        aggregate['torch_multi_vram_total_MB_sum'] = vinfo['total_bytes']/1e6
                        aggregate['torch_multi_vram_used_MB_sum'] = vinfo['used_bytes']/1e6
                        aggregate['torch_multi_vram_used_pct'] = (vinfo['used_bytes']/vinfo['total_bytes']*100.0) if vinfo['total_bytes'] else 0.0
                printed += 1
                any_ok = True
            elif be == 'cupy':
                # Mono-GPU
                vals = []
                for i in range(args.repeats):
                    s1 = bench_cupy(args.duration, 0, args.size, args.verbose)
                    vals.append(s1)
                    if args.verbose:
                        print(f"[cupy mono] run {i+1}/{args.repeats}: {s1:.3f}")
                avg, std = calc_stats(vals)
                display_result(be, 'mono', 1, args.duration, avg, std, len(vals))
                aggregate['cupy_mono_avg'] = avg
                aggregate['cupy_mono_std'] = std
                vinfo = getattr(bench_cupy, 'last_vram', None)
                if vinfo:
                    aggregate['cupy_mono_vram_total_MB'] = vinfo['total_bytes']/1e6
                    aggregate['cupy_mono_vram_used_MB'] = vinfo['used_bytes']/1e6
                    aggregate['cupy_mono_vram_used_pct'] = (vinfo['used_bytes']/vinfo['total_bytes']*100.0) if vinfo['total_bytes'] else 0.0
                printed += 1
                any_ok = True
                # Multi-GPU
                devs = list_devices_cupy()
                threads_count = len(devs) if len(devs) > 1 else 1
                vals = []
                for i in range(args.repeats):
                    if len(devs) > 1:
                        s = bench_cupy_multi(args.duration, devs, args.size, args.verbose)
                    else:
                        s = bench_cupy(args.duration, 0, args.size, args.verbose)
                    vals.append(s)
                    if args.verbose:
                        print(f"[cupy multi] run {i+1}/{args.repeats}: {s:.3f}")
                avg, std = calc_stats(vals)
                display_result(be, 'multi', threads_count, args.duration, avg, std, len(vals))
                aggregate['cupy_multi_avg'] = avg
                aggregate['cupy_multi_std'] = std
                aggregate['cupy_multi_gpus'] = threads_count
                if threads_count > 1:
                    vinfo_m = getattr(bench_cupy_multi, 'last_vram', None)
                    if vinfo_m:
                        total_sum = vinfo_m.get('total_bytes_sum') or 0
                        used_sum = vinfo_m.get('used_bytes_sum') or 0
                        aggregate['cupy_multi_vram_total_MB_sum'] = total_sum/1e6
                        aggregate['cupy_multi_vram_used_MB_sum'] = used_sum/1e6
                        aggregate['cupy_multi_vram_used_pct'] = (used_sum/total_sum*100.0) if total_sum else 0.0
                else:
                    vinfo = getattr(bench_cupy, 'last_vram', None)
                    if vinfo:
                        aggregate['cupy_multi_vram_total_MB_sum'] = vinfo['total_bytes']/1e6
                        aggregate['cupy_multi_vram_used_MB_sum'] = vinfo['used_bytes']/1e6
                        aggregate['cupy_multi_vram_used_pct'] = (vinfo['used_bytes']/vinfo['total_bytes']*100.0) if vinfo['total_bytes'] else 0.0
                printed += 1
                any_ok = True
            elif be == 'numba':
                # Mono-GPU
                vals = []
                for i in range(args.repeats):
                    s1 = bench_numba(args.duration, 0, args.size, args.verbose)
                    vals.append(s1)
                    if args.verbose:
                        print(f"[numba mono] run {i+1}/{args.repeats}: {s1:.3f}")
                avg, std = calc_stats(vals)
                display_result(be, 'mono', 1, args.duration, avg, std, len(vals))
                aggregate['numba_mono_avg'] = avg
                aggregate['numba_mono_std'] = std
                vinfo = getattr(bench_numba, 'last_vram', None)
                if vinfo:
                    aggregate['numba_mono_vram_total_MB'] = vinfo['total_bytes']/1e6
                    aggregate['numba_mono_vram_used_MB'] = vinfo['used_bytes']/1e6
                    aggregate['numba_mono_vram_used_pct'] = (vinfo['used_bytes']/vinfo['total_bytes']*100.0) if vinfo['total_bytes'] else 0.0
                printed += 1
                any_ok = True
                # Multi-GPU (fallback séquentiel)
                devs = list_devices_numba()
                threads_count = len(devs) if len(devs) > 1 else 1
                vals = []
                for i in range(args.repeats):
                    if len(devs) > 1:
                        s = 0.0
                        for d in devs:
                            s += bench_numba(args.duration, d, args.size, args.verbose)
                    else:
                        s = bench_numba(args.duration, 0, args.size, args.verbose)
                    vals.append(s)
                    if args.verbose:
                        print(f"[numba multi] run {i+1}/{args.repeats}: {s:.3f}")
                avg, std = calc_stats(vals)
                display_result(be, 'multi', threads_count, args.duration, avg, std, len(vals))
                aggregate['numba_multi_avg'] = avg
                aggregate['numba_multi_std'] = std
                aggregate['numba_multi_gpus'] = threads_count
                vinfo = getattr(bench_numba, 'last_vram', None)
                if vinfo:
                    if threads_count > 1:
                        total_sum = vinfo['total_bytes'] * threads_count
                        used_sum = vinfo['used_bytes'] * threads_count
                    else:
                        total_sum = vinfo['total_bytes']
                        used_sum = vinfo['used_bytes']
                    aggregate['numba_multi_vram_total_MB_sum'] = total_sum/1e6
                    aggregate['numba_multi_vram_used_MB_sum'] = used_sum/1e6
                    aggregate['numba_multi_vram_used_pct'] = (used_sum/total_sum*100.0) if total_sum else 0.0
                printed += 1
                any_ok = True
            
            else:
                raise RuntimeError('backend inconnu')

        except Exception as e:
            last_err = e
            if args.verbose:
                print(f"[warn] backend {be} indisponible/échec: {e}", file=sys.stderr)
            continue

    if any_ok:
        write_csv_row(aggregate)
        return 0
    else:
        print("Aucun backend GPU Python disponible (torch/cupy/numba)", file=sys.stderr)
        if last_err and args.verbose:
            print(f"Dernière erreur: {last_err}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
