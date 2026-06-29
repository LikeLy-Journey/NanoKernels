# Matrix transpose in Triton, correctness + bandwidth benchmark vs torch.
# Run: python transpose_triton.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))

import torch
import triton
import triton.language as tl
from bench import bench, report, check_close


@triton.jit
def transpose_kernel(in_ptr, out_ptr, rows, cols,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)  # over rows of input
    pid_n = tl.program_id(1)  # over cols of input
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # input row idx
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # input col idx
    # load in[rm, rn] : shape [BLOCK_M, BLOCK_N], row-major stride (cols, 1)
    in_off = rm[:, None] * cols + rn[None, :]
    mask = (rm[:, None] < rows) & (rn[None, :] < cols)
    x = tl.load(in_ptr + in_off, mask=mask)
    # store transposed into out[rn, rm] : out is [cols, rows], stride (rows, 1)
    out_off = rn[:, None] * rows + rm[None, :]
    omask = (rn[:, None] < cols) & (rm[None, :] < rows)
    tl.store(out_ptr + out_off, tl.trans(x), mask=omask)


def triton_transpose(x, BLOCK_M=32, BLOCK_N=32):
    rows, cols = x.shape
    out = torch.empty((cols, rows), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(rows, BLOCK_M), triton.cdiv(cols, BLOCK_N))
    transpose_kernel[grid](x, out, rows, cols, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out


def main():
    rows, cols = 4096, 4096
    dev = "cuda"
    x = torch.randn((rows, cols), device=dev, dtype=torch.float32)

    out = triton_transpose(x)
    ref = x.t().contiguous()
    check_close(out, ref, name="triton transpose")

    bytes_moved = 2 * rows * cols * 4  # 1 read + 1 write, fp32

    ms_triton = bench(lambda: triton_transpose(x))
    # torch .t() is a view (free); force materialization for fair compare
    ms_torch = bench(lambda: x.t().contiguous())

    report("triton transpose", ms_triton, bytes_moved=bytes_moved, ref_ms=ms_torch)
    report("torch  transpose (ref)", ms_torch, bytes_moved=bytes_moved)


if __name__ == "__main__":
    main()
