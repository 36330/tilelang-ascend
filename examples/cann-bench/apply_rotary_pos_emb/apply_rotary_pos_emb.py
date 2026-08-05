import torch
import tilelang
from tilelang import language as T

PASS_CONFIGS_FLAT = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}
PASS_CONFIGS_TILE = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_VEC_NUM = 2
_CAST_LOW2HIGH = "CAST_NONE"
_CAST_HIGH2LOW = "CAST_RINT"

DTYPE_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}

_flat_kernel_cache = {}
_tile_kernel_cache = {}


def _flat_choose_block_M(D, dtype, mode):
    UB_LIMIT = 180000
    need_cast = dtype in ("float16", "bfloat16")
    if mode == "half":
        if need_cast:
            max_sub = max(2, UB_LIMIT // (28 * D))
        else:
            max_sub = max(2, UB_LIMIT // (16 * D))
    else:
        if need_cast:
            max_sub = max(2, UB_LIMIT // (50 * D))
        else:
            max_sub = max(2, UB_LIMIT // (42 * D))
    block_M = min(max_sub, 128) * _VEC_NUM
    block_M = max(block_M, 4)
    return block_M


def _expand_cos_sin_view(cos, sin, mode):
    """Expand (S, D/2) → (S, D) using pure view ops (expand+reshape).
    No ACL op triggered — no .to(), .cat(), .repeat(), .stack().
    """
    if mode == "half":
        cos_full = cos.unsqueeze(1).expand(-1, 2, -1).reshape(cos.shape[0], -1)
        sin_full = sin.unsqueeze(1).expand(-1, 2, -1).reshape(sin.shape[0], -1)
    else:
        cos_full = cos.unsqueeze(-1).expand(-1, -1, 2).reshape(cos.shape[0], -1)
        sin_full = sin.unsqueeze(-1).expand(-1, -1, 2).reshape(sin.shape[0], -1)
    return cos_full, sin_full


# ===========================================================================
# Flat kernel — half mode (split-compute, no gather)
# ===========================================================================
def _make_flat_kernel_half(M, D, CS_DIM, N, layout, dtype, block_M):
    D_HALF = D // 2
    sub_block_M = block_M // _VEC_NUM
    m_num = T.ceildiv(M, block_M)
    need_cast = dtype in ("float16", "bfloat16")
    cal_dtype = "float32" if need_cast else dtype
    count = sub_block_M * D_HALF
    can_preload = (2 * CS_DIM * D_HALF * 4) <= 65536

    @T.prim_func
    def main(
        Q: T.Tensor((M, D), dtype),
        K: T.Tensor((M, D), dtype),
        COS: T.Tensor((CS_DIM, D_HALF), dtype),
        SIN: T.Tensor((CS_DIM, D_HALF), dtype),
        Q_OUT: T.Tensor((M, D), dtype),
        K_OUT: T.Tensor((M, D), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_start = cid * block_M + vid * sub_block_M
            with T.Scope("V"):
                cos_ub = T.alloc_ub((sub_block_M, D_HALF), cal_dtype)
                sin_ub = T.alloc_ub((sub_block_M, D_HALF), cal_dtype)
                x_first = T.alloc_ub((sub_block_M, D_HALF), cal_dtype)
                x_second = T.alloc_ub((sub_block_M, D_HALF), cal_dtype)
                y_first = T.alloc_ub((sub_block_M, D_HALF), cal_dtype)
                y_second = T.alloc_ub((sub_block_M, D_HALF), cal_dtype)
                tmp = T.alloc_ub((sub_block_M, D_HALF), cal_dtype)
                cos_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                sin_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                x_first_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                x_second_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                y_first_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                y_second_h = T.alloc_ub((sub_block_M, D_HALF), dtype)

                if can_preload:
                    cos_preload = T.alloc_ub((CS_DIM, D_HALF), cal_dtype)
                    sin_preload = T.alloc_ub((CS_DIM, D_HALF), cal_dtype)
                    if need_cast:
                        cos_pre_h = T.alloc_ub((CS_DIM, D_HALF), dtype)
                        sin_pre_h = T.alloc_ub((CS_DIM, D_HALF), dtype)
                        T.copy(COS, cos_pre_h)
                        T.copy(SIN, sin_pre_h)
                        T.tile.cast(cos_preload, cos_pre_h, _CAST_LOW2HIGH, CS_DIM * D_HALF)
                        T.tile.cast(sin_preload, sin_pre_h, _CAST_LOW2HIGH, CS_DIM * D_HALF)
                    else:
                        T.copy(COS, cos_preload)
                        T.copy(SIN, sin_preload)
                    for i in T.serial(sub_block_M):
                        row = row_start + i
                        if layout == 0:
                            T.copy(cos_preload[(row // N) % CS_DIM, :], cos_ub[i, :])
                            T.copy(sin_preload[(row // N) % CS_DIM, :], sin_ub[i, :])
                        else:
                            T.copy(cos_preload[row % CS_DIM, :], cos_ub[i, :])
                            T.copy(sin_preload[row % CS_DIM, :], sin_ub[i, :])
                else:
                    if need_cast:
                        for i in T.serial(sub_block_M):
                            row = row_start + i
                            if layout == 0:
                                T.copy(COS[(row // N) % CS_DIM, :], cos_h[i, :])
                                T.copy(SIN[(row // N) % CS_DIM, :], sin_h[i, :])
                            else:
                                T.copy(COS[row % CS_DIM, :], cos_h[i, :])
                                T.copy(SIN[row % CS_DIM, :], sin_h[i, :])
                        T.tile.cast(cos_ub, cos_h, _CAST_LOW2HIGH, count)
                        T.tile.cast(sin_ub, sin_h, _CAST_LOW2HIGH, count)
                    else:
                        for i in T.serial(sub_block_M):
                            row = row_start + i
                            if layout == 0:
                                T.copy(COS[(row // N) % CS_DIM, :], cos_ub[i, :])
                                T.copy(SIN[(row // N) % CS_DIM, :], sin_ub[i, :])
                            else:
                                T.copy(COS[row % CS_DIM, :], cos_ub[i, :])
                                T.copy(SIN[row % CS_DIM, :], sin_ub[i, :])

                if need_cast:
                    T.copy(Q[row_start, 0], x_first_h)
                    T.copy(Q[row_start, D_HALF], x_second_h)
                    T.tile.cast(x_first, x_first_h, _CAST_LOW2HIGH, count)
                    T.tile.cast(x_second, x_second_h, _CAST_LOW2HIGH, count)
                    T.tile.mul(y_first, x_first, cos_ub)
                    T.tile.mul(tmp, x_second, sin_ub)
                    T.tile.sub(y_first, y_first, tmp)
                    T.tile.mul(y_second, x_second, cos_ub)
                    T.tile.mul(tmp, x_first, sin_ub)
                    T.tile.add(y_second, y_second, tmp)
                    T.tile.cast(y_first_h, y_first, _CAST_HIGH2LOW, count)
                    T.tile.cast(y_second_h, y_second, _CAST_HIGH2LOW, count)
                    T.copy(y_first_h, Q_OUT[row_start, 0])
                    T.copy(y_second_h, Q_OUT[row_start, D_HALF])
                    T.copy(K[row_start, 0], x_first_h)
                    T.copy(K[row_start, D_HALF], x_second_h)
                    T.tile.cast(x_first, x_first_h, _CAST_LOW2HIGH, count)
                    T.tile.cast(x_second, x_second_h, _CAST_LOW2HIGH, count)
                    T.tile.mul(y_first, x_first, cos_ub)
                    T.tile.mul(tmp, x_second, sin_ub)
                    T.tile.sub(y_first, y_first, tmp)
                    T.tile.mul(y_second, x_second, cos_ub)
                    T.tile.mul(tmp, x_first, sin_ub)
                    T.tile.add(y_second, y_second, tmp)
                    T.tile.cast(y_first_h, y_first, _CAST_HIGH2LOW, count)
                    T.tile.cast(y_second_h, y_second, _CAST_HIGH2LOW, count)
                    T.copy(y_first_h, K_OUT[row_start, 0])
                    T.copy(y_second_h, K_OUT[row_start, D_HALF])
                else:
                    T.copy(Q[row_start, 0], x_first)
                    T.copy(Q[row_start, D_HALF], x_second)
                    T.tile.mul(y_first, x_first, cos_ub)
                    T.tile.mul(tmp, x_second, sin_ub)
                    T.tile.sub(y_first, y_first, tmp)
                    T.tile.mul(y_second, x_second, cos_ub)
                    T.tile.mul(tmp, x_first, sin_ub)
                    T.tile.add(y_second, y_second, tmp)
                    T.copy(y_first, Q_OUT[row_start, 0])
                    T.copy(y_second, Q_OUT[row_start, D_HALF])
                    T.copy(K[row_start, 0], x_first)
                    T.copy(K[row_start, D_HALF], x_second)
                    T.tile.mul(y_first, x_first, cos_ub)
                    T.tile.mul(tmp, x_second, sin_ub)
                    T.tile.sub(y_first, y_first, tmp)
                    T.tile.mul(y_second, x_second, cos_ub)
                    T.tile.mul(tmp, x_first, sin_ub)
                    T.tile.add(y_second, y_second, tmp)
                    T.copy(y_first, K_OUT[row_start, 0])
                    T.copy(y_second, K_OUT[row_start, D_HALF])
    return main

_flat_kernel_half_jit = tilelang.jit(out_idx=[4, 5], pass_configs=PASS_CONFIGS_FLAT)(_make_flat_kernel_half)


# ===========================================================================
# Flat kernel — interleaved mode (pre-expanded cos/sin, gather+sin_sign)
# ===========================================================================
def _make_flat_kernel_interleaved(M, D, CS_DIM, N, layout, dtype, block_M):
    """Interleaved: receives PRE-EXPANDED (CS_DIM, D) cos/sin via expand().reshape().
    No kernel-internal expansion — just load, cast, apply sin_sign, gather+compute.
    """
    sub_block_M = block_M // _VEC_NUM
    m_num = T.ceildiv(M, block_M)
    need_cast = dtype in ("float16", "bfloat16")
    cal_dtype = "float32" if need_cast else dtype
    count = sub_block_M * D
    can_preload = (2 * CS_DIM * D * 4) <= 65536

    @T.prim_func
    def main(
        Q: T.Tensor((M, D), dtype),
        K: T.Tensor((M, D), dtype),
        COS: T.Tensor((CS_DIM, D), dtype),
        SIN: T.Tensor((CS_DIM, D), dtype),
        Q_OUT: T.Tensor((M, D), dtype),
        K_OUT: T.Tensor((M, D), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_start = cid * block_M + vid * sub_block_M
            with T.Scope("V"):
                cos_ub = T.alloc_ub((sub_block_M, D), cal_dtype)
                sin_ub = T.alloc_ub((sub_block_M, D), cal_dtype)
                sin_load = T.alloc_ub((sub_block_M, D), cal_dtype)
                x = T.alloc_ub((sub_block_M, D), cal_dtype)
                x_rotate = T.alloc_ub((sub_block_M, D), cal_dtype)
                out = T.alloc_ub((sub_block_M, D), cal_dtype)

                # gather mask: swap pairs [1,0,3,2,...]
                idx_i32 = T.alloc_ub((sub_block_M, D), "int32")
                T.tile.createvecindex(idx_i32, 0)
                idx_i16 = T.alloc_ub((sub_block_M, D), "int16")
                T.copy(idx_i32, idx_i16)
                ones_i16 = T.alloc_ub((sub_block_M, D), "int16")
                T.tile.fill(ones_i16, 1)
                mask_i16 = T.alloc_ub((sub_block_M, D), "int16")
                T.tile.bitwise_xor(mask_i16, idx_i16, ones_i16)
                mask_f32 = T.alloc_ub((sub_block_M, D), "float32")
                T.copy(mask_i16, mask_f32)
                mask_i32 = T.alloc_ub((sub_block_M, D), "int32")
                T.copy(mask_f32, mask_i32)
                T.tile.mul(mask_i32, mask_i32, 4)
                gather_mask = T.alloc_ub((sub_block_M, D), "uint32")
                T.reinterpretcast(gather_mask, mask_i32, "uint32_t")

                # sin_sign: [-1,1,-1,1,...]
                sin_sign = T.alloc_ub((D,), cal_dtype)
                T.tile.fill(sin_sign, -1.0)
                D_HALF = D // 2
                for j in T.unroll(D_HALF):
                    sin_sign[2 * j + 1] = 1.0
                sin_sign_bc = T.alloc_ub((sub_block_M, D), cal_dtype)
                T.tile.broadcast(sin_sign_bc, sin_sign)

                cos_h = T.alloc_ub((sub_block_M, D), dtype)
                sin_h = T.alloc_ub((sub_block_M, D), dtype)
                x_h = T.alloc_ub((sub_block_M, D), dtype)
                out_h = T.alloc_ub((sub_block_M, D), dtype)

                # Load pre-expanded cos/sin per-row
                if can_preload:
                    cos_preload = T.alloc_ub((CS_DIM, D), cal_dtype)
                    sin_preload = T.alloc_ub((CS_DIM, D), cal_dtype)
                    if need_cast:
                        cos_pre_h = T.alloc_ub((CS_DIM, D), dtype)
                        sin_pre_h = T.alloc_ub((CS_DIM, D), dtype)
                        T.copy(COS, cos_pre_h)
                        T.copy(SIN, sin_pre_h)
                        T.tile.cast(cos_preload, cos_pre_h, _CAST_LOW2HIGH, CS_DIM * D)
                        T.tile.cast(sin_preload, sin_pre_h, _CAST_LOW2HIGH, CS_DIM * D)
                    else:
                        T.copy(COS, cos_preload)
                        T.copy(SIN, sin_preload)
                    for i in T.serial(sub_block_M):
                        row = row_start + i
                        if layout == 0:
                            T.copy(cos_preload[(row // N) % CS_DIM, :], cos_ub[i, :])
                            T.copy(sin_preload[(row // N) % CS_DIM, :], sin_load[i, :])
                        else:
                            T.copy(cos_preload[row % CS_DIM, :], cos_ub[i, :])
                            T.copy(sin_preload[row % CS_DIM, :], sin_load[i, :])
                else:
                    if need_cast:
                        for i in T.serial(sub_block_M):
                            row = row_start + i
                            if layout == 0:
                                T.copy(COS[(row // N) % CS_DIM, :], cos_h[i, :])
                                T.copy(SIN[(row // N) % CS_DIM, :], sin_h[i, :])
                            else:
                                T.copy(COS[row % CS_DIM, :], cos_h[i, :])
                                T.copy(SIN[row % CS_DIM, :], sin_h[i, :])
                        T.tile.cast(cos_ub, cos_h, _CAST_LOW2HIGH, count)
                        T.tile.cast(sin_load, sin_h, _CAST_LOW2HIGH, count)
                    else:
                        for i in T.serial(sub_block_M):
                            row = row_start + i
                            if layout == 0:
                                T.copy(COS[(row // N) % CS_DIM, :], cos_ub[i, :])
                                T.copy(SIN[(row // N) % CS_DIM, :], sin_load[i, :])
                            else:
                                T.copy(COS[row % CS_DIM, :], cos_ub[i, :])
                                T.copy(SIN[row % CS_DIM, :], sin_load[i, :])

                # sin_ub = sin_load * sin_sign
                T.tile.mul(sin_ub, sin_load, sin_sign_bc)

                # Q: gather + mul + add
                if need_cast:
                    T.copy(Q[row_start, 0], x_h)
                    T.tile.cast(x, x_h, _CAST_LOW2HIGH, count)
                    T.tile.gather(x_rotate, x, gather_mask, 0)
                    T.tile.mul(out, x, cos_ub)
                    T.tile.mul(x_rotate, x_rotate, sin_ub)
                    T.tile.add(out, out, x_rotate)
                    T.tile.cast(out_h, out, _CAST_HIGH2LOW, count)
                    T.copy(out_h, Q_OUT[row_start, 0])
                    T.copy(K[row_start, 0], x_h)
                    T.tile.cast(x, x_h, _CAST_LOW2HIGH, count)
                    T.tile.gather(x_rotate, x, gather_mask, 0)
                    T.tile.mul(out, x, cos_ub)
                    T.tile.mul(x_rotate, x_rotate, sin_ub)
                    T.tile.add(out, out, x_rotate)
                    T.tile.cast(out_h, out, _CAST_HIGH2LOW, count)
                    T.copy(out_h, K_OUT[row_start, 0])
                else:
                    T.copy(Q[row_start, 0], x)
                    T.tile.gather(x_rotate, x, gather_mask, 0)
                    T.tile.mul(out, x, cos_ub)
                    T.tile.mul(x_rotate, x_rotate, sin_ub)
                    T.tile.add(out, out, x_rotate)
                    T.copy(out, Q_OUT[row_start, 0])
                    T.copy(K[row_start, 0], x)
                    T.tile.gather(x_rotate, x, gather_mask, 0)
                    T.tile.mul(out, x, cos_ub)
                    T.tile.mul(x_rotate, x_rotate, sin_ub)
                    T.tile.add(out, out, x_rotate)
                    T.copy(out, K_OUT[row_start, 0])
    return main

_flat_kernel_interleaved_jit = tilelang.jit(out_idx=[4, 5], pass_configs=PASS_CONFIGS_FLAT)(_make_flat_kernel_interleaved)


# ===========================================================================
# Tile kernel — receives PRE-EXPANDED (S, D) cos/sin, no internal expansion
# ===========================================================================
def _make_tile_kernel(B, S, N, D, layout, mode, dtype, Block_M, S_TILE, D_TILE, num_stages):
    HalfD = D // 2
    compute_dtype = "float32"
    need_cast = dtype != compute_dtype
    elem_bytes = 4
    BN_total = B * N
    m_num = (BN_total + Block_M - 1) // Block_M
    s_num = (S + S_TILE - 1) // S_TILE
    D_TILE = D
    d_num = 1
    tile_num = s_num * d_num
    Dim1 = N if layout == 1 else S
    Dim2 = S if layout == 1 else N

    @T.prim_func
    def kernel(
        x_in_q: T.Tensor([B, Dim1, Dim2, D], dtype),
        x_in_k: T.Tensor([B, Dim1, Dim2, D], dtype),
        cos_full: T.Tensor([S, D], dtype),
        sin_full: T.Tensor([S, D], dtype),
        x_out_q: T.Tensor([B, Dim1, Dim2, D], dtype),
        x_out_k: T.Tensor([B, Dim1, Dim2, D], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                cos_ub = T.alloc_ub((2, S_TILE, D_TILE), "float32")
                sin_ub = T.alloc_ub((2, S_TILE, D_TILE), "float32")
                sin_load = T.alloc_ub((2, S_TILE, D_TILE), "float32")
                cos_h = T.alloc_ub((2, S_TILE, D_TILE), dtype)
                sin_h = T.alloc_ub((2, S_TILE, D_TILE), dtype)
                x_tile = T.alloc_ub((2, S_TILE, D_TILE), dtype)
                out_tile = T.alloc_ub((2, S_TILE, D_TILE), dtype)
                x_fp32 = T.alloc_ub((S_TILE, D_TILE), "float32")
                x_rotate = T.alloc_ub((S_TILE, D_TILE), "float32")
                out_fp32 = T.alloc_ub((S_TILE, D_TILE), "float32")

                # sin_sign
                sin_sign = T.alloc_ub((D_TILE,), "float32")
                if mode == "half":
                    T.tile.fill(sin_sign, 1.0)
                    for i in T.serial(HalfD):
                        sin_sign[i] = -1.0
                else:
                    T.tile.fill(sin_sign, -1.0)
                    for i in T.unroll(HalfD):
                        sin_sign[2 * i + 1] = 1.0
                sin_sign_bc = T.alloc_ub((S_TILE, D_TILE), "float32")
                T.tile.broadcast(sin_sign_bc, sin_sign)

                # gather rotate_mask
                idx_i32 = T.alloc_ub((S_TILE, D_TILE), "int32")
                T.tile.createvecindex(idx_i32, 0)
                idx_i16 = T.alloc_ub((S_TILE, D_TILE), "int16")
                T.copy(idx_i32, idx_i16)
                half_i16 = T.alloc_ub((S_TILE, D_TILE), "int16")
                if mode == "half":
                    T.tile.fill(half_i16, HalfD)
                else:
                    T.tile.fill(half_i16, 1)
                mask_i16 = T.alloc_ub((S_TILE, D_TILE), "int16")
                T.tile.bitwise_xor(mask_i16, idx_i16, half_i16)
                mask_f32 = T.alloc_ub((S_TILE, D_TILE), "float32")
                T.copy(mask_i16, mask_f32)
                mask_i32 = T.alloc_ub((S_TILE, D_TILE), "int32")
                T.copy(mask_f32, mask_i32)
                T.tile.mul(mask_i32, mask_i32, elem_bytes)
                rotate_mask_ub = T.alloc_ub((S_TILE, D_TILE), "uint32")
                T.reinterpretcast(rotate_mask_ub, mask_i32, "uint32_t")

                for bn_local in T.serial(Block_M):
                    bn_idx = cid * Block_M + bn_local
                    if bn_idx < BN_total:
                        b_idx = bn_idx // N
                        n_idx = bn_idx % N

                        T.set_flag("mte3", "mte2", 0)
                        T.set_flag("mte3", "mte2", 1)
                        T.wait_flag("mte3", "mte2", 0)

                        s_base_0 = 0
                        s_end_0 = T.min(S_TILE, S)
                        T.copy(cos_full[s_base_0:s_end_0, :], cos_h[0, :, :])
                        T.copy(sin_full[s_base_0:s_end_0, :], sin_h[0, :, :])
                        if vid == 0:
                            if layout == 0:
                                T.copy(x_in_q[b_idx, s_base_0:s_end_0, n_idx, :], x_tile[0, :, :])
                            else:
                                T.copy(x_in_q[b_idx, n_idx, s_base_0:s_end_0, :], x_tile[0, :, :])
                        else:
                            if layout == 0:
                                T.copy(x_in_k[b_idx, s_base_0:s_end_0, n_idx, :], x_tile[0, :, :])
                            else:
                                T.copy(x_in_k[b_idx, n_idx, s_base_0:s_end_0, :], x_tile[0, :, :])
                        T.set_flag("mte2", "v", 0)

                        for tile_id in T.serial(tile_num):
                            cur = tile_id % 2
                            nxt = (tile_id + 1) % 2
                            cur_s_base = tile_id * S_TILE
                            cur_s_end = cur_s_base + T.min(S_TILE, S - cur_s_base)

                            if tile_id + 1 < tile_num:
                                T.wait_flag("mte3", "mte2", nxt)
                                next_s_base = (tile_id + 1) * S_TILE
                                next_s_end = next_s_base + T.min(S_TILE, S - next_s_base)
                                T.copy(cos_full[next_s_base:next_s_end, :], cos_h[nxt, :, :])
                                T.copy(sin_full[next_s_base:next_s_end, :], sin_h[nxt, :, :])
                                if vid == 0:
                                    if layout == 0:
                                        T.copy(x_in_q[b_idx, next_s_base:next_s_end, n_idx, :], x_tile[nxt, :, :])
                                    else:
                                        T.copy(x_in_q[b_idx, n_idx, next_s_base:next_s_end, :], x_tile[nxt, :, :])
                                else:
                                    if layout == 0:
                                        T.copy(x_in_k[b_idx, next_s_base:next_s_end, n_idx, :], x_tile[nxt, :, :])
                                    else:
                                        T.copy(x_in_k[b_idx, n_idx, next_s_base:next_s_end, :], x_tile[nxt, :, :])
                                T.set_flag("mte2", "v", nxt)

                            T.wait_flag("mte2", "v", cur)
                            # Cast cos/sin to fp32 + apply sin_sign
                            if need_cast:
                                T.tile.cast(cos_ub[cur, :, :], cos_h[cur, :, :], "CAST_NONE", S_TILE * D_TILE)
                                T.tile.cast(sin_load[cur, :, :], sin_h[cur, :, :], "CAST_NONE", S_TILE * D_TILE)
                            else:
                                T.copy(cos_h[cur, :, :], cos_ub[cur, :, :])
                                T.copy(sin_h[cur, :, :], sin_load[cur, :, :])
                            T.tile.mul(sin_ub[cur, :, :], sin_load[cur, :, :], sin_sign_bc)

                            if vid == 0:
                                if need_cast:
                                    T.tile.cast(x_fp32, x_tile[cur, :, :], "CAST_NONE", S_TILE * D_TILE)
                                else:
                                    T.copy(x_tile[cur, :, :], x_fp32)
                                T.tile.gather(x_rotate, x_fp32, rotate_mask_ub, 0)
                                T.tile.mul(out_fp32, x_fp32, cos_ub[cur, :, :])
                                T.tile.mul_add_dst(out_fp32, x_rotate, sin_ub[cur, :, :])
                                if need_cast:
                                    T.tile.cast(out_tile[cur, :, :], out_fp32, "CAST_RINT", S_TILE * D_TILE)
                                else:
                                    T.copy(out_fp32, out_tile[cur, :, :])
                            else:
                                if need_cast:
                                    T.tile.cast(x_fp32, x_tile[cur, :, :], "CAST_NONE", S_TILE * D_TILE)
                                else:
                                    T.copy(x_tile[cur, :, :], x_fp32)
                                T.tile.gather(x_rotate, x_fp32, rotate_mask_ub, 0)
                                T.tile.mul(out_fp32, x_fp32, cos_ub[cur, :, :])
                                T.tile.mul_add_dst(out_fp32, x_rotate, sin_ub[cur, :, :])
                                if need_cast:
                                    T.tile.cast(out_tile[cur, :, :], out_fp32, "CAST_RINT", S_TILE * D_TILE)
                                else:
                                    T.copy(out_fp32, out_tile[cur, :, :])
                            T.set_flag("v", "mte3", cur)

                            T.wait_flag("v", "mte3", cur)
                            if vid == 0:
                                if layout == 0:
                                    T.copy(out_tile[cur, :, :], x_out_q[b_idx, cur_s_base:cur_s_end, n_idx, :])
                                else:
                                    T.copy(out_tile[cur, :, :], x_out_q[b_idx, n_idx, cur_s_base:cur_s_end, :])
                            else:
                                if layout == 0:
                                    T.copy(out_tile[cur, :, :], x_out_k[b_idx, cur_s_base:cur_s_end, n_idx, :])
                                else:
                                    T.copy(out_tile[cur, :, :], x_out_k[b_idx, n_idx, cur_s_base:cur_s_end, :])
                            T.set_flag("mte3", "mte2", cur)

                        T.wait_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 1)
    return kernel

_tile_kernel_jit = tilelang.jit(pass_configs=PASS_CONFIGS_TILE)(_make_tile_kernel)


# ===========================================================================
# Python entry
# ===========================================================================
def apply_rotary_pos_emb_flat(query, key, cos, sin, layout=0, rotaryMode="half"):
    if layout == 0:
        B, S, N, D = query.shape
    else:
        B, N, S, D = query.shape
    M = B * S * N if layout == 0 else B * N * S
    tl_dtype = DTYPE_STR[query.dtype]

    q_2d = query.reshape(M, D)
    k_2d = key.reshape(M, D)

    if rotaryMode == "half":
        # half: pass raw (CS_DIM, D_HALF) — kernel does split-compute
        if cos.dim() == 3:
            cos_2d = cos.reshape(-1, cos.shape[-1])
            sin_2d = sin.reshape(-1, sin.shape[-1])
            cs_dim = B * S
        else:
            cos_2d = cos
            sin_2d = sin
            cs_dim = S
    else:
        # interleaved: pre-expand (CS_DIM, D_HALF) → (CS_DIM, D) via expand().reshape()
        if cos.dim() == 3:
            cos_raw = cos.reshape(-1, cos.shape[-1])
            sin_raw = sin.reshape(-1, sin.shape[-1])
        else:
            cos_raw = cos
            sin_raw = sin
        cos_2d, sin_2d = _expand_cos_sin_view(cos_raw, sin_raw, rotaryMode)
        cs_dim = cos_2d.shape[0]

    block_M = _flat_choose_block_M(D, tl_dtype, rotaryMode)
    cache_key = (M, D, cs_dim, N, layout, block_M, tl_dtype, rotaryMode)
    if cache_key not in _flat_kernel_cache:
        if rotaryMode == "half":
            _flat_kernel_cache[cache_key] = _flat_kernel_half_jit(M, D, cs_dim, N, layout, tl_dtype, block_M)
        else:
            _flat_kernel_cache[cache_key] = _flat_kernel_interleaved_jit(M, D, cs_dim, N, layout, tl_dtype, block_M)
    kernel = _flat_kernel_cache[cache_key]

    q_out_2d, k_out_2d = kernel(q_2d, k_2d, cos_2d, sin_2d)
    q_out = q_out_2d.reshape(query.shape)
    k_out = k_out_2d.reshape(key.shape)
    return q_out, k_out


def apply_rotary_pos_emb_tl(query, key, cos, sin, layout=0, rotaryMode="half"):
    if layout == 0:
        B, S, N, D = query.shape
    else:
        B, N, S, D = query.shape

    dtype_str = DTYPE_STR[query.dtype]
    BN_total = B * N
    M_total = B * N * S

    use_flat = (BN_total < 20) or (S <= 31) or (M_total <= 50000)

    if use_flat:
        return apply_rotary_pos_emb_flat(query, key, cos, sin, layout, rotaryMode)

    # tile kernel — pre-expand cos/sin via expand().reshape() (pure view, no ACL op)
    if cos.dim() == 3:
        cos_raw = cos.reshape(-1, cos.shape[-1])
        sin_raw = sin.reshape(-1, sin.shape[-1])
    else:
        cos_raw = cos
        sin_raw = sin
    cos_full, sin_full = _expand_cos_sin_view(cos_raw, sin_raw, rotaryMode)

    UB_LIMIT = 180224
    ds = 4 if dtype_str == "float32" else 2
    s_tile_max = (UB_LIMIT - 8 * D) // (D * (72 + 2 * ds))
    S_TILE = min(max(s_tile_max, 1), S)

    if BN_total < 20:
        Block_M = 1
    else:
        Block_M = min(BN_total, max(3, BN_total // 20))
    num_stages = 2
    D_TILE = D

    tile_cache_key = (B, S, N, D, layout, rotaryMode, dtype_str, Block_M, S_TILE, D_TILE, num_stages)
    if tile_cache_key not in _tile_kernel_cache:
        _tile_kernel_cache[tile_cache_key] = _tile_kernel_jit(
            B, S, N, D, layout, rotaryMode, dtype_str, Block_M, S_TILE, D_TILE, num_stages,
        )
    kernel = _tile_kernel_cache[tile_cache_key]

    q_out = torch.empty_like(query)
    k_out = torch.empty_like(key)
    kernel(query, key, cos_full, sin_full, q_out, k_out)
    return q_out, k_out


def apply_rotary_pos_emb(query, key, cos, sin, layout=0, rotaryMode="half"):
    return apply_rotary_pos_emb_tl(query, key, cos, sin, layout, rotaryMode)


__all__ = ["apply_rotary_pos_emb"]
