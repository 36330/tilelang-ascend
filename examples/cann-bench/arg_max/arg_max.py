import os
import sys
import torch
import torch_npu  # noqa: F401
import tilelang
from tilelang import language as T

try:
    from ._common import torch_dtype_to_tl
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import torch_dtype_to_tl

_kernel_cache = {}

_ALIGN = 8
_MAX_BLOCK_N = 2048
_VEC_NUM = 2
_BLOCK_M = 32
_SUB_BLOCK_M = _BLOCK_M // _VEC_NUM
_NONLAST_BLOCK_N = 128

_SYNC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _get_pad_val(in_dtype):
    if in_dtype in ["int32", "int64"]:
        return -2147483647
    else:
        return -T.infinity("float32")


@tilelang.jit(out_idx=[1], pass_configs=_SYNC)
def _cast_int64_to_float32_kernel(M, N, block_N):
    m_num = T.ceildiv(M, _BLOCK_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor([M, N], "int64"),
        B: T.Tensor([M, N], "float32"),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_tile = T.alloc_ub([block_N], "int64")
            a_tmp = T.alloc_ub([block_N], "int32")
            b_tile = T.alloc_ub([block_N], "float32")

            for ri in T.serial(_SUB_BLOCK_M):
                row = cid * _BLOCK_M + vid * _SUB_BLOCK_M + ri
                if row < M:
                    for bn in T.serial(n_num):
                        col_start = bn * block_N
                        T.copy(
                            A[row, col_start : col_start + block_N],
                            a_tile,
                            pad_value=-2147483647,
                        )
                        T.tile.cast(a_tmp, a_tile, "CAST_NONE", block_N)
                        T.tile.cast(b_tile, a_tmp, "CAST_NONE", block_N)
                        T.copy(b_tile, B[row, col_start : col_start + block_N])

    return main


@tilelang.jit(out_idx=[1], pass_configs=_SYNC)
def _argmax_nonlast_kernel(batch, M, N, block_N, in_dtype):
    cal_dtype = "float32"
    use_fp32_compute = in_dtype in ["float16", "bfloat16", "int32"]
    has_nan = in_dtype in ["float16", "bfloat16", "float32"]
    pad_val = _get_pad_val(in_dtype)
    n_num = T.ceildiv(N, block_N)
    total_tasks = batch * n_num

    @T.prim_func
    def main(
        A: T.Tensor([batch, M, N], in_dtype),
        Out: T.Tensor([batch, N], "int64"),
    ):
        with T.Kernel(T.ceildiv(total_tasks, _VEC_NUM), is_npu=True) as (cid, vid):
            task = cid * _VEC_NUM + vid
            if task < total_tasks:
                b = task // n_num
                bi = task % n_num
                col_start = bi * block_N

                running_max = T.alloc_ub([block_N], cal_dtype)
                running_idx = T.alloc_ub([block_N], cal_dtype)
                idx_buf = T.alloc_ub([block_N], cal_dtype)
                gt_mask = T.alloc_ub([block_N], cal_dtype)
                row_tile = T.alloc_ub([block_N], in_dtype)
                row_cal = T.alloc_ub([block_N], cal_dtype)

                T.tile.fill(running_max, -T.infinity(cal_dtype))
                T.tile.fill(running_idx, 0.0)

                for m in T.serial(M):
                    T.copy(A[b, m, col_start], row_tile, pad_value=pad_val)
                    if use_fp32_compute:
                        T.tile.cast(row_cal, row_tile, "CAST_NONE", block_N)
                    else:
                        T.copy(row_tile, row_cal)

                    if has_nan:
                        T.tile.compare(gt_mask, row_cal, row_cal, "EQ")
                        T.tile.select(
                            row_cal,
                            gt_mask,
                            row_cal,
                            T.infinity(cal_dtype),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )

                    T.tile.fill(idx_buf, T.cast(m, cal_dtype))
                    T.tile.compare(gt_mask, row_cal, running_max, "GT")
                    T.tile.select(
                        running_max,
                        gt_mask,
                        row_cal,
                        running_max,
                        "VSEL_TENSOR_TENSOR_MODE",
                    )
                    T.tile.select(
                        running_idx,
                        gt_mask,
                        idx_buf,
                        running_idx,
                        "VSEL_TENSOR_TENSOR_MODE",
                    )

                idx_out = T.alloc_ub([block_N], "int64")
                T.tile.cast(idx_out, running_idx, "CAST_RINT", block_N)
                T.copy(idx_out, Out[b, col_start])

    return main


@tilelang.jit(out_idx=[1], pass_configs=_SYNC)
def _argmax_wholereduce_kernel(M, N, block_N, in_dtype="float16"):
    cal_dtype = "float32"
    use_fp32_compute = in_dtype in ["float16", "bfloat16", "int32"]
    has_nan = in_dtype in ["float16", "bfloat16", "float32"]
    pad_val = _get_pad_val(in_dtype)
    m_num = T.ceildiv(M, _BLOCK_M)

    @T.prim_func
    def main(
        A: T.Tensor([M, N], in_dtype),
        Out: T.Tensor([M], "int64"),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_tile = T.alloc_ub([block_N], in_dtype)
            a_cal = T.alloc_ub([block_N], cal_dtype)
            idx_buf = T.alloc_ub([block_N], cal_dtype)
            mask = T.alloc_ub([block_N], cal_dtype)
            sel = T.alloc_ub([block_N], cal_dtype)
            max_val = T.alloc_ub([1], cal_dtype)
            min_idx = T.alloc_ub([1], cal_dtype)
            idx_out_i64 = T.alloc_ub([8], "int64")

            T.tile.createvecindex(idx_buf, 0)

            for ri in T.serial(_SUB_BLOCK_M):
                row = cid * _BLOCK_M + vid * _SUB_BLOCK_M + ri
                if row < M:
                    T.copy(A[row, 0:block_N], a_tile, pad_value=pad_val)

                    if use_fp32_compute:
                        T.tile.cast(a_cal, a_tile, "CAST_NONE", block_N)
                    else:
                        T.copy(a_tile, a_cal)

                    if has_nan:
                        T.tile.compare(mask, a_cal, a_cal, "EQ")
                        T.tile.select(
                            a_cal,
                            mask,
                            a_cal,
                            T.infinity(cal_dtype),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )

                    T.reduce_max(a_cal, max_val, dim=-1, clear=True)
                    T.tile.compare(mask, a_cal, max_val[0], "EQ")
                    T.tile.select(
                        sel, mask, idx_buf, 999999.0, "VSEL_TENSOR_SCALAR_MODE"
                    )
                    T.reduce_min(sel, min_idx, dim=-1, clear=True)
                    T.tile.cast(idx_out_i64, min_idx, "CAST_RINT", 8)
                    T.copy(idx_out_i64[0:1], Out[row : row + 1])

    return main


@tilelang.jit(out_idx=[1], pass_configs=_SYNC)
def _argmax_sort_kernel(M, N, block_N, in_dtype="float16"):
    cal_dtype = "float32"
    use_fp32_compute = in_dtype in ["float16", "bfloat16", "int32"]
    has_nan = in_dtype in ["float16", "bfloat16", "float32"]
    pad_val = _get_pad_val(in_dtype)
    m_num = T.ceildiv(M, _BLOCK_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor([M, N], in_dtype),
        Out: T.Tensor([M], "int64"),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            running_max = T.alloc_ub([8], cal_dtype)
            running_idx_i32 = T.alloc_ub([8], "int32")
            running_idx_i64 = T.alloc_ub([8], "int64")
            a_tile = T.alloc_ub([block_N], in_dtype)
            a_cal = T.alloc_ub([block_N], cal_dtype)
            tile_max = T.alloc_ub([1], cal_dtype)
            sort_dst = T.alloc_ub([block_N * 2], cal_dtype)
            tile_max_f = T.alloc_ub([8], cal_dtype)
            tile_idx_f = T.alloc_ub([8], cal_dtype)
            best_tile = T.alloc_ub([1], "int32")
            mask = T.alloc_ub([block_N], cal_dtype)

            for ri in T.serial(_SUB_BLOCK_M):
                row = cid * _BLOCK_M + vid * _SUB_BLOCK_M + ri
                if row < M:
                    T.tile.fill(running_max, -T.infinity(cal_dtype))
                    T.tile.fill(running_idx_i32, 0)
                    T.tile.fill(best_tile, 0)

                    for bn in T.serial(n_num):
                        col_start = bn * block_N
                        T.copy(
                            A[row, col_start : col_start + block_N],
                            a_tile,
                            pad_value=pad_val,
                        )
                        if use_fp32_compute:
                            T.tile.cast(a_cal, a_tile, "CAST_NONE", block_N)
                        else:
                            T.copy(a_tile, a_cal)

                        if has_nan:
                            T.tile.compare(mask, a_cal, a_cal, "EQ")
                            T.tile.select(
                                a_cal,
                                mask,
                                a_cal,
                                T.infinity(cal_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                        T.reduce_max(a_cal, tile_max, dim=-1, clear=True)
                        if tile_max[0] > running_max[0]:
                            running_max[0] = tile_max[0]
                            best_tile[0] = bn

                    col_start = best_tile[0] * block_N
                    T.copy(
                        A[row, col_start : col_start + block_N],
                        a_tile,
                        pad_value=pad_val,
                    )
                    if use_fp32_compute:
                        T.tile.cast(a_cal, a_tile, "CAST_NONE", block_N)
                    else:
                        T.copy(a_tile, a_cal)

                    if has_nan:
                        T.tile.compare(mask, a_cal, a_cal, "EQ")
                        T.tile.select(
                            a_cal,
                            mask,
                            a_cal,
                            T.infinity(cal_dtype),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )

                    T.tile.sort(sort_dst, a_cal, block_N)
                    T.tile.gather_mask(tile_max_f, sort_dst, "P0101")
                    T.tile.gather_mask(tile_idx_f, sort_dst, "P1010")
                    running_idx_i32[0] = col_start + T.cast(tile_idx_f[0], "int32")
                    T.tile.cast(running_idx_i64, running_idx_i32, "CAST_NONE", 8)
                    T.copy(running_idx_i64[0:1], Out[row : row + 1])

    return main


def _get_kernel(M, N, tl_dtype):
    key = (M, N, tl_dtype)
    if key not in _kernel_cache:
        block_N = min(((N + 63) // 64) * 64, _MAX_BLOCK_N)
        if block_N < 64:
            block_N = 64
        n_num = (N + block_N - 1) // block_N
        if n_num <= 1:
            _kernel_cache[key] = _argmax_wholereduce_kernel(
                M, N, block_N, in_dtype=tl_dtype
            )
        else:
            _kernel_cache[key] = _argmax_sort_kernel(M, N, block_N, in_dtype=tl_dtype)
    return _kernel_cache[key]


def _get_nonlast_kernel(batch, M, N, tl_dtype):
    key = ("nonlast", batch, M, N, tl_dtype)
    if key not in _kernel_cache:
        block_N = min(
            ((N + _NONLAST_BLOCK_N - 1) // _NONLAST_BLOCK_N) * _NONLAST_BLOCK_N,
            _MAX_BLOCK_N,
        )
        if block_N < _NONLAST_BLOCK_N:
            block_N = _NONLAST_BLOCK_N
        _kernel_cache[key] = _argmax_nonlast_kernel(
            batch, M, N, block_N, in_dtype=tl_dtype
        )
    return _kernel_cache[key]


def _get_cast_int64_kernel(M, N):
    key = ("cast_int64", M, N)
    if key not in _kernel_cache:
        block_N = min(((N + 63) // 64) * 64, _MAX_BLOCK_N)
        if block_N < 64:
            block_N = 64
        _kernel_cache[key] = _cast_int64_to_float32_kernel(M, N, block_N)
    return _kernel_cache[key]


def arg_max(input: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    ndim = input.ndim
    if ndim == 0:
        return torch.tensor(0, dtype=torch.int64, device=input.device)
    dim = dim % ndim
    original_shape = list(input.shape)
    is_int64 = input.dtype == torch.int64

    if dim != ndim - 1:
        reduce_size = input.shape[dim]
        non_reduce_shape = [input.shape[i] for i in range(ndim) if i != dim]
        outer_size = 1
        for s in non_reduce_shape:
            outer_size *= s
        batch = 1
        for i in range(dim):
            batch *= input.shape[i]
        inner_size = 1
        for i in range(dim + 1, ndim):
            inner_size *= input.shape[i]

        x_3d = input.reshape(batch, reduce_size, inner_size)
        if is_int64:
            x_3d = _get_cast_int64_kernel(batch * reduce_size, inner_size)(
                x_3d.reshape(batch * reduce_size, inner_size)
            ).reshape(batch, reduce_size, inner_size)
            tl_dtype = "float"
        else:
            tl_dtype = torch_dtype_to_tl(input.dtype)
        kernel = _get_nonlast_kernel(batch, reduce_size, inner_size, tl_dtype)
        out_2d = kernel(x_3d)
    else:
        x = input if input.is_contiguous() else input.contiguous()
        N = x.shape[-1]
        outer = 1
        for s in x.shape[:-1]:
            outer *= s
        x_2d = x.reshape(outer, N)
        if is_int64:
            x_2d = _get_cast_int64_kernel(outer, N)(x_2d)
            tl_dtype = "float"
        else:
            tl_dtype = torch_dtype_to_tl(input.dtype)
        kernel = _get_kernel(outer, N, tl_dtype)
        out_2d = kernel(x_2d)

    if keepdim:
        if dim != ndim - 1:
            out_shape = non_reduce_shape + [1]
            out = out_2d.reshape(out_shape)
            full_keepdim = list(original_shape)
            full_keepdim[dim] = 1
            out = out.reshape(full_keepdim)
        else:
            transposed_keepdim_shape = list(x.shape[:-1]) + [1]
            out = out_2d.reshape(transposed_keepdim_shape)
            final_shape = list(original_shape)
            final_shape[dim] = 1
            out = out.reshape(final_shape)
    else:
        if dim != ndim - 1:
            out = out_2d.reshape(non_reduce_shape)
        else:
            out = out_2d.reshape(x.shape[:-1])

    return out


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.float32).npu()
    y = arg_max(x, dim=-1)
    torch.npu.synchronize()
    print("Done")
