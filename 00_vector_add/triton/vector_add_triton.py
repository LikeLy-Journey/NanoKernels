# Vector add in Triton, with correctness check and bandwidth benchmark vs torch.
# Run: python vector_add_triton.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))

import torch
import triton
import triton.language as tl
from bench import bench, gbps, report, check_close


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x, y, BLOCK=1024):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


def main():
    n = 1 << 24  # 16M
    dev = "cuda"
    x = torch.randn(n, device=dev, dtype=torch.float32)
    y = torch.randn(n, device=dev, dtype=torch.float32)

    out = triton_add(x, y)
    ref = x + y
    check_close(out, ref, name="triton vector_add")

    bytes_moved = 3 * n * 4  # 2 read + 1 write, fp32

    ms_triton = bench(lambda: triton_add(x, y))
    ms_torch = bench(lambda: x + y)

    report("triton add", ms_triton, bytes_moved=bytes_moved, ref_ms=ms_torch)
    report("torch  add (ref)", ms_torch, bytes_moved=bytes_moved)


if __name__ == "__main__":
    main()
