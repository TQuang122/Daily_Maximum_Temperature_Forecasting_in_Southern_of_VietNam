# utils/performance_monitor.py
import os, gc
from time import perf_counter
import tracemalloc

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:
    _PROC = None

def _mb(b): 
    return b / (1024**2)

def timed_mem_call(fn, *args, label="", **kwargs):
    """
    Chạy fn(*args, **kwargs) và trả về (result, stats_dict).
    stats_dict: seconds, py_peak_mb, và nếu có psutil thì thêm
    rss_before_mb, rss_after_mb, rss_delta_mb.
    """
    gc.collect()

    rss_before = _PROC.memory_info().rss if _PROC else None

    tracemalloc.start()
    t0 = perf_counter()
    result = fn(*args, **kwargs)
    elapsed = perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after = _PROC.memory_info().rss if _PROC else None
    rss_delta = (rss_after - rss_before) if (_PROC and rss_after is not None and rss_before is not None) else None

    msg = f"[{label}] time={elapsed:.2f}s | Py-peak={_mb(peak_bytes):.1f} MB"
    if _PROC:
        msg += f" | RSS(before/after/delta)={_mb(rss_before):.1f}/{_mb(rss_after):.1f}/{_mb(rss_delta):.1f} MB"
    print(msg)

    stats = {"seconds": elapsed, "py_peak_mb": _mb(peak_bytes)}
    if _PROC:
        stats.update({
            "rss_before_mb": _mb(rss_before),
            "rss_after_mb": _mb(rss_after),
            "rss_delta_mb": _mb(rss_delta),
        })
    return result, stats