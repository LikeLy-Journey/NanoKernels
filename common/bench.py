"""Shared benchmark helpers for NanoKernels.

Provides:
- bench(fn, *, warmup, iters): returns avg milliseconds using CUDA events.
- gbps(bytes_moved, ms): effective memory bandwidth in GB/s.
- gflops(flop, ms): compute throughput in GFLOP/s.
- report(name, ms, *, bytes_moved=None, flop=None, ref_ms=None): pretty print one row.
"""
import torch


def bench(fn, *, warmup=10, iters=100):
    """Time a callable on the current CUDA stream with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def gbps(bytes_moved, ms):
    return bytes_moved / (ms * 1e-3) / 1e9


def gflops(flop, ms):
    return flop / (ms * 1e-3) / 1e9


def report(name, ms, *, bytes_moved=None, flop=None, ref_ms=None):
    line = f"{name:<28} {ms:8.4f} ms"
    if bytes_moved is not None:
        line += f" | {gbps(bytes_moved, ms):8.1f} GB/s"
    if flop is not None:
        line += f" | {gflops(flop, ms):8.1f} GFLOP/s"
    if ref_ms is not None and ms > 0:
        line += f" | {ref_ms / ms:5.2f}x vs ref"
    print(line)


def check_close(out, ref, *, rtol=1e-3, atol=1e-3, name="result"):
    ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
    max_err = (out - ref).abs().max().item() if out.numel() else 0.0
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: max_abs_err={max_err:.3e}")
    return ok
