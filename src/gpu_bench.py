#!/usr/bin/env python3
import argparse
import sys
import time
import threading
import os
import socket
from datetime import datetime
import math


def fmt_device_info(backend, name, idx):
    return f"BACKEND {backend} DEVICE_IDX {idx} DEVICE {name}"


def bench_torch(duration, device_index, N, verbose):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda non disponible")
    torch.cuda.set_device(device_index)
    dev_name = torch.cuda.get_device_name(device_index)

    # Try allocate; downscale if OOM
    xN = N
    while True:
        try:
            a = torch.rand(xN, device='cuda', dtype=torch.float32)
            b = torch.rand(xN, device='cuda', dtype=torch.float32)
            c = torch.rand(xN, device='cuda', dtype=torch.float32)
            out = torch.empty_like(a)
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and xN > (1<<18):
                xN //= 2
                torch.cuda.empty_cache()
            else:
                raise

    # Warmup
    for _ in range(3):
        out = torch.addcmul(c, a, b)  # FMA-like
        out = torch.addcmul(out, b, a)
        out = torch.addcmul(out, a, out)
    torch.cuda.synchronize()

    # Calibrate
    iters = 256
    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        out = torch.addcmul(c, a, b)
        out = torch.addcmul(out, b, a)
        out = torch.addcmul(out, a, out)
    stop_evt.record(); stop_evt.synchronize()
    ms = start_evt.elapsed_time(stop_evt)
    secs = ms / 1000.0
    if secs > 0:
        scale = duration / secs
        scale = min(max(scale, 0.5), 64.0)
        iters = max(int(iters * scale), 1)

    if verbose:
        print(fmt_device_info('torch', dev_name, device_index))
        print(f"PARAM N {xN} ITERS {iters} TARGET {duration:.3f}s")

    # Measure
    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        out = torch.addcmul(c, a, b)
        out = torch.addcmul(out, b, a)
        out = torch.addcmul(out, a, out)
    stop_evt.record(); stop_evt.synchronize()
    ms = start_evt.elapsed_time(stop_evt)

    total_flops = 6.0 * iters * xN  # 3 FMA = 6 FLOPs
    flops_per_s = total_flops / (ms / 1000.0)
    return flops_per_s


def bench_cupy(duration, device_index, N, verbose):
    import cupy as cp
    dev = cp.cuda.Device(device_index)
    dev.use()
    dev_name = dev.attributes.get('Name') or cp.cuda.runtime.getDeviceProperties(device_index)['name'].decode()

    # Try allocate; downscale if OOM
    xN = N
    while True:
        try:
            a = cp.random.rand(xN, dtype=cp.float32)
            b = cp.random.rand(xN, dtype=cp.float32)
            c = cp.random.rand(xN, dtype=cp.float32)
            out = cp.empty_like(a)
            break
        except cp.cuda.memory.OutOfMemoryError:
            if xN > (1<<18):
                xN //= 2
                cp.get_default_memory_pool().free_all_blocks()
            else:
                raise

    kernel = cp.ElementwiseKernel(
        in_params='float32 a, float32 b, float32 c, int32 iters',
        out_params='float32 out',
        operation='''
            float x=a, y=b, z=c;
            for (int i=0;i<iters;++i) {
              x = fmaf(x,y,z);
              y = fmaf(y,z,x);
              z = fmaf(z,x,y);
            }
            out = x + y + z;''',
        name='fma_loop')

    # Warmup
    kernel(a, b, c, 1, out)
    cp.cuda.Stream.null.synchronize()

    # Calibrate
    iters = 256
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record(); kernel(a, b, c, iters, out); end.record(); end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end)
    secs = ms / 1000.0
    if secs > 0:
        scale = duration / secs
        scale = min(max(scale, 0.5), 64.0)
        iters = max(int(iters * scale), 1)

    if verbose:
        print(fmt_device_info('cupy', dev_name, device_index))
        print(f"PARAM N {xN} ITERS {iters} TARGET {duration:.3f}s")

    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record(); kernel(a, b, c, iters, out); end.record(); end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end)

    total_flops = 6.0 * iters * xN
    flops_per_s = total_flops / (ms / 1000.0)
    return flops_per_s


def bench_numba(duration, device_index, N, verbose):
    from numba import cuda
    cuda.select_device(device_index)
    dev = cuda.get_current_device()
    dev_name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name

    import numpy as np
    xN = N
    while True:
        try:
            a = np.random.rand(xN).astype(np.float32)
            b = np.random.rand(xN).astype(np.float32)
            c = np.random.rand(xN).astype(np.float32)
            d_a = cuda.to_device(a)
            d_b = cuda.to_device(b)
            d_c = cuda.to_device(c)
            d_out = cuda.device_array_like(d_a)
            break
        except cuda.cudadrv.driver.CudaAPIError:
            if xN > (1<<18):
                xN //= 2
            else:
                raise

    @cuda.jit
    def fma_loop(a, b, c, out, iters):
        i = cuda.grid(1)
        if i >= a.size:
            return
        x = a[i]; y = b[i]; z = c[i]
        for _ in range(iters):
            x = x*y + z
            y = y*z + x
            z = z*x + y
        out[i] = x + y + z

    threads = 256
    blocks = (xN + threads - 1) // threads
    # Warmup
    fma_loop[blocks, threads](d_a, d_b, d_c, d_out, 1); cuda.synchronize()

    iters = 256
    t0 = time.perf_counter(); fma_loop[blocks, threads](d_a, d_b, d_c, d_out, iters); cuda.synchronize(); t1 = time.perf_counter()
    secs = t1 - t0
    if secs > 0:
        scale = duration / secs
        scale = min(max(scale, 0.5), 64.0)
        iters = max(int(iters * scale), 1)

    if verbose:
        print(fmt_device_info('numba', dev_name, device_index))
        print(f"PARAM N {xN} ITERS {iters} TARGET {duration:.3f}s")

    t0 = time.perf_counter(); fma_loop[blocks, threads](d_a, d_b, d_c, d_out, iters); cuda.synchronize(); t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0
    total_flops = 6.0 * iters * xN
    flops_per_s = total_flops / (ms / 1000.0)
    return flops_per_s


def bench_opencl(duration, device_index, N, verbose):
    import pyopencl as cl
    import numpy as np
    platforms = cl.get_platforms()
    gpus = []
    for p in platforms:
        for d in p.get_devices(device_type=cl.device_type.GPU):
            gpus.append(d)
    if not gpus:
        raise RuntimeError("Aucun GPU OpenCL détecté")
    device = gpus[device_index % len(gpus)]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    dev_name = device.name

    xN = N
    while True:
        try:
            a = np.random.rand(xN).astype(np.float32)
            b = np.random.rand(xN).astype(np.float32)
            c = np.random.rand(xN).astype(np.float32)
            mf = cl.mem_flags
            d_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
            d_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
            d_c = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
            d_out = cl.Buffer(ctx, mf.WRITE_ONLY, size=a.nbytes)
            break
        except cl.MemoryError:
            if xN > (1<<18):
                xN //= 2
            else:
                raise

    prg = cl.Program(ctx, r"""
    __kernel void fma_loop(__global const float* a, __global const float* b,
                           __global const float* c, __global float* out,
                           int iters) {
        int i = get_global_id(0);
        float x = a[i], y = b[i], z = c[i];
        for (int k=0; k<iters; ++k) {
            x = mad(x,y,z);
            y = mad(y,z,x);
            z = mad(z,x,y);
        }
        out[i] = x + y + z;
    }
    """).build()

    # Warmup
    evt = prg.fma_loop(queue, (xN,), None, d_a, d_b, d_c, d_out, np.int32(1)); evt.wait()

    iters = 256
    evt = prg.fma_loop(queue, (xN,), None, d_a, d_b, d_c, d_out, np.int32(iters)); evt.wait()
    ms = (evt.profile.end - evt.profile.start) / 1e6
    secs = ms / 1000.0
    if secs > 0:
        scale = duration / secs
        scale = min(max(scale, 0.5), 64.0)
        iters = max(int(iters * scale), 1)

    if verbose:
        print(fmt_device_info('opencl', dev_name, device_index))
        print(f"PARAM N {xN} ITERS {iters} TARGET {duration:.3f}s")

    evt = prg.fma_loop(queue, (xN,), None, d_a, d_b, d_c, d_out, np.int32(iters)); evt.wait()
    ms = (evt.profile.end - evt.profile.start) / 1e6
    total_flops = 6.0 * iters * xN
    flops_per_s = total_flops / (ms / 1000.0)
    return flops_per_s


def list_devices_torch():
    import torch
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def list_devices_cupy():
    import cupy as cp
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception:
        return []
    return list(range(n))


def list_devices_numba():
    from numba import cuda
    try:
        return [i.id for i in cuda.gpus]
    except Exception:
        return []


def list_devices_opencl():
    try:
        import pyopencl as cl
    except Exception:
        return []
    devs = []
    try:
        for p in cl.get_platforms():
            for d in p.get_devices(device_type=cl.device_type.GPU):
                devs.append(d)
    except Exception:
        return []
    return list(range(len(devs)))


def bench_torch_multi(duration, device_indices, N, verbose):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda non disponible")
    if not device_indices:
        raise RuntimeError("Aucun GPU torch")

    # Calibre sur le premier GPU
    iters = 256
    torch.cuda.set_device(device_indices[0])
    a = torch.rand(N, device='cuda', dtype=torch.float32)
    b = torch.rand(N, device='cuda', dtype=torch.float32)
    c = torch.rand(N, device='cuda', dtype=torch.float32)
    out = torch.empty_like(a)
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        out = torch.addcmul(c, a, b)
        out = torch.addcmul(out, b, a)
        out = torch.addcmul(out, a, out)
    e.record(); e.synchronize()
    ms = s.elapsed_time(e)
    secs = ms/1000.0
    if secs>0:
        scale = duration/secs
        scale = min(max(scale,0.5),64.0)
        iters = max(int(iters*scale),1)

    # Lance en parallèle sur chaque GPU
    totals = {}
    times_ms = {}
    def worker(idx):
        torch.cuda.set_device(idx)
        a = torch.rand(N, device='cuda', dtype=torch.float32)
        b = torch.rand(N, device='cuda', dtype=torch.float32)
        c = torch.rand(N, device='cuda', dtype=torch.float32)
        out = torch.empty_like(a)
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            out = torch.addcmul(c, a, b)
            out = torch.addcmul(out, b, a)
            out = torch.addcmul(out, a, out)
        e.record(); e.synchronize()
        ms = s.elapsed_time(e)
        totals[idx] = 6.0 * iters * N
        times_ms[idx] = ms

    threads = [threading.Thread(target=worker, args=(i,)) for i in device_indices]
    for t in threads: t.start()
    for t in threads: t.join()

    if verbose:
        names = [torch.cuda.get_device_name(i) for i in device_indices]
        print(f"BACKEND torch DEVICES {device_indices} NAMES {names} ITERS {iters} TARGET {duration:.3f}s")

    total_flops = sum(totals.values())
    max_time = max(times_ms.values())/1000.0
    return total_flops / max_time


def bench_cupy_multi(duration, device_indices, N, verbose):
    import cupy as cp
    if not device_indices:
        raise RuntimeError("Aucun GPU cupy")
    # Calibre sur le premier
    iters = 256
    dev0 = cp.cuda.Device(device_indices[0]); dev0.use()
    a = cp.random.rand(N, dtype=cp.float32)
    b = cp.random.rand(N, dtype=cp.float32)
    c = cp.random.rand(N, dtype=cp.float32)
    out = cp.empty_like(a)
    kernel = cp.ElementwiseKernel(
        'float32 a,float32 b,float32 c,int32 iters','float32 out',
        'float x=a,y=b,z=c; for(int i=0;i<iters;++i){x=fmaf(x,y,z); y=fmaf(y,z,x); z=fmaf(z,x,y);} out=x+y+z;','fma_loop')
    s=cp.cuda.Event(); e=cp.cuda.Event(); s.record(); kernel(a,b,c,iters,out); e.record(); e.synchronize();
    ms = cp.cuda.get_elapsed_time(s,e)
    secs = ms/1000.0
    if secs>0:
        scale = duration/secs
        scale = min(max(scale,0.5),64.0)
        iters = max(int(iters*scale),1)

    totals = {}
    times_ms = {}
    def worker(idx):
        dev = cp.cuda.Device(idx); dev.use()
        a = cp.random.rand(N, dtype=cp.float32)
        b = cp.random.rand(N, dtype=cp.float32)
        c = cp.random.rand(N, dtype=cp.float32)
        out = cp.empty_like(a)
        s=cp.cuda.Event(); e=cp.cuda.Event(); s.record(); kernel(a,b,c,iters,out); e.record(); e.synchronize();
        times_ms[idx] = cp.cuda.get_elapsed_time(s,e)
        totals[idx] = 6.0 * iters * N

    threads = [threading.Thread(target=worker, args=(i,)) for i in device_indices]
    for t in threads: t.start()
    for t in threads: t.join()

    if verbose:
        names = [cp.cuda.runtime.getDeviceProperties(i)['name'].decode() for i in device_indices]
        print(f"BACKEND cupy DEVICES {device_indices} NAMES {names} ITERS {iters} TARGET {duration:.3f}s")

    total_flops = sum(totals.values())
    max_time = max(times_ms.values())/1000.0
    return total_flops / max_time


def main():
    p = argparse.ArgumentParser(description='GPU compute benchmark (Python). Exécute toujours mono et multi pour chaque backend disponible.')
    p.add_argument('--duration', type=float, default=2.0, help='durée cible en secondes')
    p.add_argument('--size', type=int, default=1<<23, help='taille du vecteur (peut être réduit si OOM)')
    p.add_argument('--repeats', type=int, default=3, help='nombre de répétitions pour moyenne/écart-type')
    p.add_argument('--verbose', action='store_true')
    # outputs sous la racine du projet (parent de src)
    p.add_argument('--csv-dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs'), help='répertoire pour stocker le CSV consolidé')
    p.add_argument('--node', type=str, default=socket.gethostname().split('.')[0], help='nom du nœud pour les CSV')
    args = p.parse_args()

    backends = ['torch','cupy','numba','opencl']

    # Prépare CSV dir
    csv_dir = args.csv_dir
    os.makedirs(csv_dir, exist_ok=True)

    def write_csv_row(results):
        """Écrit une seule ligne récapitulative dans gpu_<node>.csv.
        Colonnes: node,runs,duration_s,timestamp,(<be>_mono_avg,<be>_mono_std,<be>_multi_avg,<be>_multi_std,<be>_multi_gpus)*
        """
        path = os.path.join(csv_dir, f"gpu_{args.node}.csv")
        be_cols = []
        for be in backends:
            be_cols += [
                f"{be}_mono_avg", f"{be}_mono_std",
                f"{be}_multi_avg", f"{be}_multi_std", f"{be}_multi_gpus"
            ]
        header = 'node,runs,duration_s,timestamp,' + ','.join(be_cols) + '\n'
        if not os.path.exists(path):
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
                print('THREADS 1'); print(f'DURATION {args.duration:.3f}'); print(f'SCORE {avg:.3f}'); print(f'STD {std:.3f}'); print(f'RUNS {len(vals)}')
                aggregate['torch_mono_avg'] = avg
                aggregate['torch_mono_std'] = std
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
                print(f'THREADS {threads_count}'); print(f'DURATION {args.duration:.3f}'); print(f'SCORE {avg:.3f}'); print(f'STD {std:.3f}'); print(f'RUNS {len(vals)}')
                aggregate['torch_multi_avg'] = avg
                aggregate['torch_multi_std'] = std
                aggregate['torch_multi_gpus'] = threads_count
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
                print('THREADS 1'); print(f'DURATION {args.duration:.3f}'); print(f'SCORE {avg:.3f}'); print(f'STD {std:.3f}'); print(f'RUNS {len(vals)}')
                aggregate['cupy_mono_avg'] = avg
                aggregate['cupy_mono_std'] = std
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
                print(f'THREADS {threads_count}'); print(f'DURATION {args.duration:.3f}'); print(f'SCORE {avg:.3f}'); print(f'STD {std:.3f}'); print(f'RUNS {len(vals)}')
                aggregate['cupy_multi_avg'] = avg
                aggregate['cupy_multi_std'] = std
                aggregate['cupy_multi_gpus'] = threads_count
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
                print('THREADS 1'); print(f'DURATION {args.duration:.3f}'); print(f'SCORE {avg:.3f}'); print(f'STD {std:.3f}'); print(f'RUNS {len(vals)}')
                aggregate['numba_mono_avg'] = avg
                aggregate['numba_mono_std'] = std
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
                print(f'THREADS {threads_count}'); print(f'DURATION {args.duration:.3f}'); print(f'SCORE {avg:.3f}'); print(f'STD {std:.3f}'); print(f'RUNS {len(vals)}')
                aggregate['numba_multi_avg'] = avg
                aggregate['numba_multi_std'] = std
                aggregate['numba_multi_gpus'] = threads_count
                printed += 1
                any_ok = True
            elif be == 'opencl':
                # Mono-GPU
                vals = []
                for i in range(args.repeats):
                    s1 = bench_opencl(args.duration, 0, args.size, args.verbose)
                    vals.append(s1)
                    if args.verbose:
                        print(f"[opencl mono] run {i+1}/{args.repeats}: {s1:.3f}")
                avg, std = calc_stats(vals)
                print('THREADS 1'); print(f'DURATION {args.duration:.3f}'); print(f'SCORE {avg:.3f}'); print(f'STD {std:.3f}'); print(f'RUNS {len(vals)}')
                aggregate['opencl_mono_avg'] = avg
                aggregate['opencl_mono_std'] = std
                printed += 1
                any_ok = True
                # Multi-GPU (fallback séquentiel)
                devs = list_devices_opencl()
                threads_count = len(devs) if len(devs) > 1 else 1
                vals = []
                for i in range(args.repeats):
                    if len(devs) > 1:
                        s = 0.0
                        for d in devs:
                            s += bench_opencl(args.duration, d, args.size, args.verbose)
                    else:
                        s = bench_opencl(args.duration, 0, args.size, args.verbose)
                    vals.append(s)
                    if args.verbose:
                        print(f"[opencl multi] run {i+1}/{args.repeats}: {s:.3f}")
                avg, std = calc_stats(vals)
                print(f'THREADS {threads_count}'); print(f'DURATION {args.duration:.3f}'); print(f'SCORE {avg:.3f}'); print(f'STD {std:.3f}'); print(f'RUNS {len(vals)}')
                aggregate['opencl_multi_avg'] = avg
                aggregate['opencl_multi_std'] = std
                aggregate['opencl_multi_gpus'] = threads_count
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
        print("Aucun backend GPU Python disponible (torch/cupy/numba/pyopencl)", file=sys.stderr)
        if last_err and args.verbose:
            print(f"Dernière erreur: {last_err}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
