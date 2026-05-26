from __future__ import annotations

try:
    import torch
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised in CPU-only environments.
    torch = None
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

        for k_start in range(0, k, block_k):
            k_mask = k_start + offs_k
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < m) & (k_mask[None, :] < k), other=0.0)
            b = tl.load(b_ptrs, mask=(k_mask[:, None] < k) & (offs_n[None, :] < n), other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += block_k * stride_ak
            b_ptrs += block_k * stride_bk

        c = accumulator.to(tl.float16)
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        weight_ptr,
        y_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / n_cols
        y = x * tl.rsqrt(variance + eps) * weight
        tl.store(y_ptr + row * n_cols + offsets, y, mask=mask)


def triton_matmul(a, b):
    _require_cuda_stack()
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("triton_matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul shapes are incompatible")
    a = a.contiguous()
    b = b.contiguous()
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    block_m = 16
    block_n = 16
    block_k = 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        block_m,
        block_n,
        block_k,
    )
    return c


def torch_rmsnorm(x, weight, eps: float = 1e-6):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps) * weight).to(x.dtype)


def triton_rmsnorm(x, weight, eps: float = 1e-6):
    _require_cuda_stack()
    if x.ndim != 2:
        raise ValueError("triton_rmsnorm expects a 2D tensor")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("rmsnorm weight must match hidden dimension")
    x = x.contiguous()
    weight = weight.contiguous()
    rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 8192:
        raise ValueError("rmsnorm kernel supports hidden sizes up to 8192")
    y = torch.empty_like(x)
    _rmsnorm_kernel[(rows,)](x, weight, y, n_cols, eps, block_size)
    return y


def _require_cuda_stack() -> None:
    if torch is None or triton is None:
        raise RuntimeError("PyTorch and Triton are required. Install with `uv sync --extra cuda`.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton kernels.")
