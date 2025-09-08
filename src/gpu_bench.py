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
            print(
                f"[error] L'environnement conda actif ('{cur}') ne correspond pas à l'environnement requis '{expected_name}'.", file=sys.stderr)
            sys.exit(3)


def main():
    p = argparse.ArgumentParser(
        description='GPU compute benchmark (Python). Exécute toujours mono et multi pour chaque backend disponible.')
    p.add_argument('--duration', type=float, default=3.0,
                   help='durée cible en secondes')
    p.add_argument('--size', type=int, default=1 << 23,
                   help='taille du vecteur (peut être réduit si OOM)')
    p.add_argument('--repeats', type=int, default=5,
                   help='nombre de répétitions pour moyenne/écart-type')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--conda-env', type=str, default=None,
                   help="nom de l'environnement conda requis (obligatoire: un conda actif doit être présent)")
    # outputs sous la racine du projet (parent de src)
    p.add_argument('--csv-dir', type=str, default=os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'outputs'), help='répertoire pour stocker le CSV consolidé')
    p.add_argument('--node', type=str, default=socket.gethostname().split('.')
                   [0], help='nom du nœud pour les CSV')
    p.add_argument('--vram-frac', type=float, default=None,
                   help='fraction VRAM cible (0.05-0.95) pour ajuster la taille des buffers GPU')
    p.add_argument('--warmup', type=int, default=None,
                   help='override du nombre d\'itérations de warmup (0..50)')
    args = p.parse_args()

    # Vérifie conda actif et (optionnellement) le nom d'env requis
    ensure_conda_active(args.conda_env)

    backends = ['torch', 'cupy', 'numba']

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

    # Nouveau format (aligné sur le CPU) mais avec backend séparé et colonnes VRAM
    # En-tête: node,backend,mode,nb_gpu,runs,duration_s,avg_events_per_s,stddev_events_per_s,min_events_per_s,max_events_per_s,vram_total_MB,vram_used_MB,vram_used_pct,timestamp
    gpu_header = 'node,backend,mode,nb_gpu,runs,duration_s,avg_events_per_s,stddev_events_per_s,min_events_per_s,max_events_per_s,vram_total_MB,vram_used_MB,vram_used_pct,heterogeneous,timestamp'
    gpu_csv_path = os.path.join(csv_dir, f"gpu_{args.node}.csv")

    def ensure_gpu_header():
        if os.path.exists(gpu_csv_path):
            try:
                with open(gpu_csv_path, 'r') as f:
                    first = f.readline().rstrip('\n')
                if first != gpu_header:
                    # sauvegarde ancien format
                    ts = datetime.now().strftime('%Y%m%d%H%M%S')
                    os.replace(gpu_csv_path, gpu_csv_path + f'.bak.{ts}')
            except Exception:
                pass
        if (not os.path.exists(gpu_csv_path)) or os.path.getsize(gpu_csv_path) == 0:
            with open(gpu_csv_path, 'w') as f:
                f.write(gpu_header + '\n')

    ensure_gpu_header()

    def write_gpu_line(backend: str, mode: str, threads: int, runs: int, duration: float,
                       avg: float, std: float, vmin: float, vmax: float,
                       vram_total: float | None, vram_used: float | None, vram_pct: float | None,
                       heterogeneous: int | None):
        ts = datetime.now().isoformat(timespec='seconds')

        def fmt(x):
            if x is None:
                return ''
            return f"{x:.3f}"
        def fmt_int(x):
            if x is None:
                return ''
            return str(int(x))
        line = (
            f"{args.node},{backend},{mode},{threads},{runs},{duration:.3f},{avg:.3f},{std:.3f},{vmin:.3f},{vmax:.3f},{fmt(vram_total)},{fmt(vram_used)},{fmt(vram_pct)},{fmt_int(heterogeneous)},{ts}\n"
        )
        with open(gpu_csv_path, 'a') as f:
            f.write(line)

    def calc_stats(vals):
        n = len(vals)
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0
        s = sum(vals)
        ss = sum(v*v for v in vals)
        m = s / n
        v = (ss / n) - (m * m)
        if v < 0:
            v = 0.0
        return m, math.sqrt(v), min(vals), max(vals)

    last_err = None
    any_ok = False
    # Plus d'agrégation multi-backend: écriture ligne par ligne
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
                        print(
                            f"[torch mono] run {i+1}/{args.repeats}: {s1:.3f}")
                avg, std, vmin, vmax = calc_stats(vals)
                display_result(be, 'mono', 1, args.duration,
                               avg, std, len(vals))
                # VRAM mono torch
                vinfo = getattr(bench_torch, 'last_vram', None)
                vram_total = vram_used = vram_pct = None
                if vinfo:
                    total = vinfo.get('total_bytes') or 0
                    used = vinfo.get('used_bytes') or 0
                    vram_total = total/1e6
                    vram_used = used/1e6
                    vram_pct = (used/total*100.0) if total else 0.0
                write_gpu_line('torch', 'mono', 1, len(
                    vals), args.duration, avg, std, vmin, vmax, vram_total, vram_used, vram_pct, 0)
                printed += 1
                any_ok = True
                # Multi-GPU
                devs = list_devices_torch()
                threads_count = len(devs) if len(devs) > 1 else 1
                vals = []
                for i in range(args.repeats):
                    if len(devs) > 1:
                        s = bench_torch_multi(
                            args.duration, devs, args.size, args.verbose)
                    else:
                        s = bench_torch(args.duration, 0,
                                        args.size, args.verbose)
                    vals.append(s)
                    if args.verbose:
                        print(
                            f"[torch multi] run {i+1}/{args.repeats}: {s:.3f}")
                avg, std, vmin, vmax = calc_stats(vals)
                display_result(be, 'multi', threads_count,
                               args.duration, avg, std, len(vals))
                # VRAM multi torch
                vram_total = vram_used = vram_pct = None
                hetero_flag = 0
                if threads_count > 1:
                    vinfo_m = getattr(bench_torch_multi, 'last_vram', None)
                    if vinfo_m:
                        total_sum = vinfo_m.get('total_bytes_sum') or 0
                        used_sum = vinfo_m.get('used_bytes_sum') or 0
                        vram_total = total_sum/1e6
                        vram_used = used_sum/1e6
                        vram_pct = (used_sum/total_sum *
                                    100.0) if total_sum else 0.0
                        per_dev = vinfo_m.get('per_device') or []
                        if per_dev:
                            totals = {d.get('total_bytes') for d in per_dev if d.get('total_bytes')}
                            if len(totals) > 1:
                                hetero_flag = 1
                else:
                    vinfo = getattr(bench_torch, 'last_vram', None)
                    if vinfo:
                        total = vinfo.get('total_bytes') or 0
                        used = vinfo.get('used_bytes') or 0
                        vram_total = total/1e6
                        vram_used = used/1e6
                        vram_pct = (used/total*100.0) if total else 0.0
                write_gpu_line('torch', 'multi', threads_count, len(
                    vals), args.duration, avg, std, vmin, vmax, vram_total, vram_used, vram_pct, hetero_flag)
                printed += 1
                any_ok = True
            elif be == 'cupy':
                # Mono-GPU
                vals = []
                for i in range(args.repeats):
                    s1 = bench_cupy(args.duration, 0, args.size, args.verbose)
                    vals.append(s1)
                    if args.verbose:
                        print(
                            f"[cupy mono] run {i+1}/{args.repeats}: {s1:.3f}")
                avg, std, vmin, vmax = calc_stats(vals)
                display_result(be, 'mono', 1, args.duration,
                               avg, std, len(vals))
                vinfo = getattr(bench_cupy, 'last_vram', None)
                vram_total = vram_used = vram_pct = None
                if vinfo:
                    total = vinfo.get('total_bytes') or 0
                    used = vinfo.get('used_bytes') or 0
                    vram_total = total/1e6
                    vram_used = used/1e6
                    vram_pct = (used/total*100.0) if total else 0.0
                write_gpu_line('cupy', 'mono', 1, len(
                    vals), args.duration, avg, std, vmin, vmax, vram_total, vram_used, vram_pct, 0)
                printed += 1
                any_ok = True
                # Multi-GPU
                devs = list_devices_cupy()
                threads_count = len(devs) if len(devs) > 1 else 1
                vals = []
                for i in range(args.repeats):
                    if len(devs) > 1:
                        s = bench_cupy_multi(
                            args.duration, devs, args.size, args.verbose)
                    else:
                        s = bench_cupy(args.duration, 0,
                                       args.size, args.verbose)
                    vals.append(s)
                    if args.verbose:
                        print(
                            f"[cupy multi] run {i+1}/{args.repeats}: {s:.3f}")
                avg, std, vmin, vmax = calc_stats(vals)
                display_result(be, 'multi', threads_count,
                               args.duration, avg, std, len(vals))
                vram_total = vram_used = vram_pct = None
                hetero_flag = 0
                if threads_count > 1:
                    vinfo_m = getattr(bench_cupy_multi, 'last_vram', None)
                    if vinfo_m:
                        total_sum = vinfo_m.get('total_bytes_sum') or 0
                        used_sum = vinfo_m.get('used_bytes_sum') or 0
                        vram_total = total_sum/1e6
                        vram_used = used_sum/1e6
                        vram_pct = (used_sum/total_sum *
                                    100.0) if total_sum else 0.0
                        per_dev = vinfo_m.get('per_device') or []
                        if per_dev:
                            totals = {d.get('total_bytes') for d in per_dev if d.get('total_bytes')}
                            if len(totals) > 1:
                                hetero_flag = 1
                else:
                    vinfo = getattr(bench_cupy, 'last_vram', None)
                    if vinfo:
                        total = vinfo.get('total_bytes') or 0
                        used = vinfo.get('used_bytes') or 0
                        vram_total = total/1e6
                        vram_used = used/1e6
                        vram_pct = (used/total*100.0) if total else 0.0
                write_gpu_line('cupy', 'multi', threads_count, len(
                    vals), args.duration, avg, std, vmin, vmax, vram_total, vram_used, vram_pct, hetero_flag)
                printed += 1
                any_ok = True
            elif be == 'numba':
                # Mono-GPU
                vals = []
                for i in range(args.repeats):
                    s1 = bench_numba(args.duration, 0, args.size, args.verbose)
                    vals.append(s1)
                    if args.verbose:
                        print(
                            f"[numba mono] run {i+1}/{args.repeats}: {s1:.3f}")
                avg, std, vmin, vmax = calc_stats(vals)
                display_result(be, 'mono', 1, args.duration,
                               avg, std, len(vals))
                vinfo = getattr(bench_numba, 'last_vram', None)
                vram_total = vram_used = vram_pct = None
                if vinfo:
                    total = vinfo.get('total_bytes') or 0
                    used = vinfo.get('used_bytes') or 0
                    vram_total = total/1e6
                    vram_used = used/1e6
                    vram_pct = (used/total*100.0) if total else 0.0
                write_gpu_line('numba', 'mono', 1, len(
                    vals), args.duration, avg, std, vmin, vmax, vram_total, vram_used, vram_pct, 0)
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
                            s += bench_numba(args.duration, d,
                                             args.size, args.verbose)
                    else:
                        s = bench_numba(args.duration, 0,
                                        args.size, args.verbose)
                    vals.append(s)
                    if args.verbose:
                        print(
                            f"[numba multi] run {i+1}/{args.repeats}: {s:.3f}")
                avg, std, vmin, vmax = calc_stats(vals)
                display_result(be, 'multi', threads_count,
                               args.duration, avg, std, len(vals))
                vram_total = vram_used = vram_pct = None
                vinfo = getattr(bench_numba, 'last_vram', None)
                if vinfo:
                    if threads_count > 1:
                        total_sum = (vinfo.get('total_bytes')
                                     or 0) * threads_count
                        used_sum = (vinfo.get('used_bytes')
                                    or 0) * threads_count
                        vram_total = total_sum/1e6
                        vram_used = used_sum/1e6
                        vram_pct = (used_sum/total_sum *
                                    100.0) if total_sum else 0.0
                    else:
                        total = vinfo.get('total_bytes') or 0
                        used = vinfo.get('used_bytes') or 0
                        vram_total = total/1e6
                        vram_used = used/1e6
                        vram_pct = (used/total*100.0) if total else 0.0
                # Fallback numba: pas de per-device détaillé -> flag 0 (ou vide). Ici 0.
                write_gpu_line('numba', 'multi', threads_count, len(
                    vals), args.duration, avg, std, vmin, vmax, vram_total, vram_used, vram_pct, 0)
                printed += 1
                any_ok = True

            else:
                raise RuntimeError('backend inconnu')

        except Exception as e:
            last_err = e
            if args.verbose:
                print(
                    f"[warn] backend {be} indisponible/échec: {e}", file=sys.stderr)
            continue

    if any_ok:
        return 0
    else:
        print("Aucun backend GPU Python disponible (torch/cupy/numba)", file=sys.stderr)
        if last_err and args.verbose:
            print(f"Dernière erreur: {last_err}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
