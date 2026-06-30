# Reduce-sum in Triton, correctness + bandwidth benchmark vs torch.
# Two-stage reduction: a grid of blocks each reduce a chunk to one partial
# (atomic-free), then a tiny second kernel sums the partials. We compare the
# whole pipeline against torch.sum. Run: python reduce_triton.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))

import torch
import triton
import triton.language as tl
from bench import bench, report, check_close


@triton.jit
def reduce_kernel(in_ptr, out_ptr, n,
                  BLOCK: tl.constexpr, ITEMS_PER_BLOCK: tl.constexpr):
    # grid-stride within a block: each block owns ITEMS_PER_BLOCK elements,
    # walked in BLOCK-wide strips, accumulated in registers, then tree-reduced.
    pid = tl.program_id(0)
    base = pid * ITEMS_PER_BLOCK
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, ITEMS_PER_BLOCK, BLOCK):
        idx = base + off + tl.arange(0, BLOCK)
        x = tl.load(in_ptr + idx, mask=idx < n, other=0.0)
        acc += x
    out_ptr_val = tl.sum(acc, axis=0)     # in-block reduction (compiler-optimized)
    tl.store(out_ptr + pid, out_ptr_val)


def triton_reduce(x, BLOCK=1024, ITEMS_PER_BLOCK=8192):
    n = x.numel()
    blocks = triton.cdiv(n, ITEMS_PER_BLOCK)
    partials = torch.empty((blocks,), device=x.device, dtype=torch.float32)
    reduce_kernel[(blocks,)](x, partials, n,
                             BLOCK=BLOCK, ITEMS_PER_BLOCK=ITEMS_PER_BLOCK)
    return partials.sum()  # tiny tail sum (blocks is small), folded into torch


def main():
    n = 1 << 24  # 16M
    dev = "cuda"
    x = torch.rand(n, device=dev, dtype=torch.float32)

    out = triton_reduce(x)
    ref = x.sum()
    check_close(out, ref, name="triton reduce", rtol=1e-3, atol=1e-1)

    bytes_moved = n * 4  # every element read once

    ms_triton = bench(lambda: triton_reduce(x))
    ms_torch = bench(lambda: x.sum())

    report("triton reduce", ms_triton, bytes_moved=bytes_moved, ref_ms=ms_torch)
    report("torch  reduce (ref)", ms_torch, bytes_moved=bytes_moved)


if __name__ == "__main__":
    main()
