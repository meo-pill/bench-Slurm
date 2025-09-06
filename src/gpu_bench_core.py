"""Fonctions de bench GPU et utilitaires de détection des devices.

Contient les kernels et exécutions unitaires (mono/multi) pour torch, cupy,
et numba (CUDA). Le support OpenCL a été retiré.
"""
import threading
import os

# Objectif d'utilisation VRAM (fraction du total). Ajuste la taille des buffers
# a,b,c,out (~4 * 4 octets * N) pour approcher cette fraction, en respectant
# une marge de sécurité et en conservant la logique de réduction si OOM.
_DEF_VRAM_TARGET = 0.80
try:
    _env_frac = float(os.environ.get("BENCH_VRAM_FRAC", ""))
    if 0.05 <= _env_frac <= 0.95:
        VRAM_TARGET_FRAC = _env_frac
    else:
        VRAM_TARGET_FRAC = _DEF_VRAM_TARGET
except Exception:
    VRAM_TARGET_FRAC = _DEF_VRAM_TARGET


def _adjust_size_for_vram(initial_N, total_bytes, free_bytes, arrays=4, dtype_bytes=4, target_frac=None):
    """Calcule une taille N ajustée pour consommer ~VRAM_TARGET_FRAC de la VRAM.

    initial_N : taille demandée par l'appelant (borne minimale)
    total_bytes, free_bytes : mémoire GPU (total / libre)
    arrays : nombre d'arrays de taille N alloués (a,b,c,out=4)
    dtype_bytes : taille en octets d'un élément (float32=4)

    target_frac = VRAM_TARGET_FRAC if target_frac is None else target_frac
    On cible min(target_frac * total, (target_frac+0.05) * free) pour ne pas
    écraser d'autres allocations résiduelles. On ne réduit jamais en-dessous d'initial_N.
    """
    if initial_N <= 0:
        base = 1 << 18  # point de départ raisonnable si l'utilisateur force N<=0
    else:
        base = initial_N
    bytes_per_set = arrays * dtype_bytes
    if total_bytes <= 0 or free_bytes <= 0:
        return base
    target_bytes = min(target_frac * total_bytes, (target_frac + 0.05) * free_bytes)
    if target_bytes <= 0:
        return base
    n_target = int(target_bytes // bytes_per_set)
    if n_target < base:
        return base
    return n_target

def set_vram_target(frac: float):
    """Permet de surcharger dynamiquement la fraction cible VRAM (0.05..0.95).
    Ignore silencieusement les valeurs hors plage.
    """
    global VRAM_TARGET_FRAC
    try:
        if 0.05 <= float(frac) <= 0.95:
            VRAM_TARGET_FRAC = float(frac)
    except Exception:
        pass


def fmt_device_info(backend, name, idx):
    return f"BACKEND {backend} DEVICE_IDX {idx} DEVICE {name}"


def bench_torch(duration, device_index, N, verbose):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda non disponible")
    torch.cuda.set_device(device_index)
    dev_name = torch.cuda.get_device_name(device_index)

    # Try allocate; downscale if OOM
    # Ajuste N pour viser ~80% VRAM si possible
    mem_info = None
    try:
        free_b, total_b = torch.cuda.mem_get_info()  # free, total
        mem_info = (free_b, total_b)
        xN = _adjust_size_for_vram(N, total_b, free_b)
    except Exception:
        xN = N
        mem_info = None
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
        if mem_info:
            free_b, total_b = mem_info
            used_bytes = 4 * 4 * xN
            frac = used_bytes / total_b if total_b else 0.0
            print(f"VRAM target={VRAM_TARGET_FRAC*100:.1f}% alloc~{frac*100:.1f}% bytes={used_bytes/1e6:.1f}MB total={total_b/1e6:.1f}MB")
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
    # Enregistre stats VRAM
    try:
        if mem_info:
            bench_torch.last_vram = {
                'used_bytes': used_bytes,
                'total_bytes': total_b,
                'N': xN,
                'arrays': 4
            }
    except Exception:
        pass
    return flops_per_s


def bench_cupy(duration, device_index, N, verbose):
    import cupy as cp
    dev = cp.cuda.Device(device_index)
    dev.use()
    dev_name = dev.attributes.get('Name') or cp.cuda.runtime.getDeviceProperties(device_index)['name'].decode()

    # Try allocate; downscale if OOM
    mem_info = None
    try:
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        mem_info = (free_b, total_b)
        xN = _adjust_size_for_vram(N, total_b, free_b)
    except Exception:
        xN = N
        mem_info = None
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
        if mem_info:
            free_b, total_b = mem_info
            used_bytes = 4 * 4 * xN
            frac = used_bytes / total_b if total_b else 0.0
            print(f"VRAM target={VRAM_TARGET_FRAC*100:.1f}% alloc~{frac*100:.1f}% bytes={used_bytes/1e6:.1f}MB total={total_b/1e6:.1f}MB")
        print(f"PARAM N {xN} ITERS {iters} TARGET {duration:.3f}s")

    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record(); kernel(a, b, c, iters, out); end.record(); end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end)

    total_flops = 6.0 * iters * xN
    flops_per_s = total_flops / (ms / 1000.0)
    try:
        if mem_info:
            bench_cupy.last_vram = {
                'used_bytes': used_bytes,
                'total_bytes': total_b,
                'N': xN,
                'arrays': 4
            }
    except Exception:
        pass
    return flops_per_s


def bench_numba(duration, device_index, N, verbose):
    from numba import cuda
    cuda.select_device(device_index)
    dev = cuda.get_current_device()
    dev_name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name

    import numpy as np
    mem_info = None
    try:
        free_b, total_b = cuda.current_context().get_memory_info()
        mem_info = (free_b, total_b)
        xN = _adjust_size_for_vram(N, total_b, free_b)
    except Exception:
        xN = N
        mem_info = None
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
    import time as _t
    t0 = _t.perf_counter(); fma_loop[blocks, threads](d_a, d_b, d_c, d_out, iters); cuda.synchronize(); t1 = _t.perf_counter()
    secs = t1 - t0
    if secs > 0:
        scale = duration / secs
        scale = min(max(scale, 0.5), 64.0)
        iters = max(int(iters * scale), 1)

    if verbose:
        print(fmt_device_info('numba', dev_name, device_index))
        if mem_info:
            free_b, total_b = mem_info
            used_bytes = 4 * 4 * xN
            frac = used_bytes / total_b if total_b else 0.0
            print(f"VRAM target={VRAM_TARGET_FRAC*100:.1f}% alloc~{frac*100:.1f}% bytes={used_bytes/1e6:.1f}MB total={total_b/1e6:.1f}MB")
        print(f"PARAM N {xN} ITERS {iters} TARGET {duration:.3f}s")

    t0 = _t.perf_counter(); fma_loop[blocks, threads](d_a, d_b, d_c, d_out, iters); cuda.synchronize(); t1 = _t.perf_counter()
    ms = (t1 - t0) * 1000.0
    total_flops = 6.0 * iters * xN
    flops_per_s = total_flops / (ms / 1000.0)
    try:
        if mem_info:
            bench_numba.last_vram = {
                'used_bytes': used_bytes,
                'total_bytes': total_b,
                'N': xN,
                'arrays': 4
            }
    except Exception:
        pass
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


def list_devices_opencl():  # Conservé pour compat rétro, renvoie toujours []
    return []


def bench_torch_multi(duration, device_indices, N, verbose):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda non disponible")
    if not device_indices:
        raise RuntimeError("Aucun GPU torch")

    torch.cuda.set_device(device_indices[0])
    # Ajuster N sur le premier device
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        xN = _adjust_size_for_vram(N, total_b, free_b)
    except Exception:
        xN = N

    # Calibrage
    iters = 256
    a = torch.rand(xN, device='cuda', dtype=torch.float32)
    b = torch.rand(xN, device='cuda', dtype=torch.float32)
    c = torch.rand(xN, device='cuda', dtype=torch.float32)
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

    totals = {}
    times_ms = {}
    def worker(idx):
        torch.cuda.set_device(idx)
        # Ajuster N pour chaque device (indépendant)
        try:
            f_b, t_b = torch.cuda.mem_get_info()
            localN = _adjust_size_for_vram(xN, t_b, f_b)
        except Exception:
            localN = xN
        a = torch.rand(localN, device='cuda', dtype=torch.float32)
        b = torch.rand(localN, device='cuda', dtype=torch.float32)
        c = torch.rand(localN, device='cuda', dtype=torch.float32)
        out = torch.empty_like(a)
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            out = torch.addcmul(c, a, b)
            out = torch.addcmul(out, b, a)
            out = torch.addcmul(out, a, out)
        e.record(); e.synchronize()
        ms = s.elapsed_time(e)
        totals[idx] = 6.0 * iters * localN
        times_ms[idx] = ms
        if verbose:
            used_bytes = 4*4*localN
            frac = used_bytes / t_b if 't_b' in locals() and t_b else 0.0
            print(f"torch dev{idx} multi VRAM target={VRAM_TARGET_FRAC*100:.1f}% alloc~{frac*100:.1f}% N={localN}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in device_indices]
    for t in threads: t.start()
    for t in threads: t.join()

    if verbose:
        names = [torch.cuda.get_device_name(i) for i in device_indices]
        print(f"BACKEND torch DEVICES {device_indices} NAMES {names} ITERS {iters} TARGET {duration:.3f}s")

    total_flops = sum(totals.values())
    max_time = max(times_ms.values())/1000.0
    # VRAM multi (somme des used / total si dispo)
    try:
        used_sum = sum(v for v in locals().get('totals_used', {}).values()) if 'totals_used' in locals() else None
    except Exception:
        used_sum = None
    # Collect inside worker? adjust worker to record used and total
    return total_flops / max_time


def bench_cupy_multi(duration, device_indices, N, verbose):
    import cupy as cp
    if not device_indices:
        raise RuntimeError("Aucun GPU cupy")
    dev0 = cp.cuda.Device(device_indices[0]); dev0.use()
    try:
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        xN = _adjust_size_for_vram(N, total_b, free_b)
    except Exception:
        xN = N
    # Calibrage
    iters = 256
    a = cp.random.rand(xN, dtype=cp.float32)
    b = cp.random.rand(xN, dtype=cp.float32)
    c = cp.random.rand(xN, dtype=cp.float32)
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
        try:
            f_b, t_b = cp.cuda.runtime.memGetInfo()
            localN = _adjust_size_for_vram(xN, t_b, f_b)
        except Exception:
            localN = xN
        a = cp.random.rand(localN, dtype=cp.float32)
        b = cp.random.rand(localN, dtype=cp.float32)
        c = cp.random.rand(localN, dtype=cp.float32)
        out = cp.empty_like(a)
        s=cp.cuda.Event(); e=cp.cuda.Event(); s.record(); kernel(a,b,c,iters,out); e.record(); e.synchronize();
        times_ms[idx] = cp.cuda.get_elapsed_time(s,e)
        totals[idx] = 6.0 * iters * localN
        if verbose:
            used_bytes = 4*4*localN
            frac = used_bytes / t_b if 't_b' in locals() and t_b else 0.0
            print(f"cupy dev{idx} multi VRAM target={VRAM_TARGET_FRAC*100:.1f}% alloc~{frac*100:.1f}% N={localN}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in device_indices]
    for t in threads: t.start()
    for t in threads: t.join()

    if verbose:
        names = [cp.cuda.runtime.getDeviceProperties(i)['name'].decode() for i in device_indices]
        print(f"BACKEND cupy DEVICES {device_indices} NAMES {names} ITERS {iters} TARGET {duration:.3f}s")

    total_flops = sum(totals.values())
    max_time = max(times_ms.values())/1000.0
    return total_flops / max_time
