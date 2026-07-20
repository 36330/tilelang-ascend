# cpliance with the License.
# # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# # See LICENSE in the root of the software repository for the full text of the License.
# # ----------------------------------------------------------------------------------------------------------

import torch

import tilelang
import tilelang.language as T


def bench_us(fn, warmup=10, repeat=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn()
    end.record()

    torch.npu.synchronize()
    return start.elapsed_time(end) / repeat * 1000.0


"""
ApplyRotaryPosEmb 算子 Torch Golden 参考实现

对 query 和 key 执行旋转位置编码 (RoPE) 计算
公式:
    rotate_half(x) = concat(-x[head_dim/2:], x[:head_dim/2])
    y = (x * cos) + (rotate_half(x) * sin)

参考:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - LLaMA: https://github.com/meta-llama/llama
    - HuggingFace transformers: https://huggingface.co/docs/transformers/internal/rope_utils
"""


def apply_rotary_pos_emb(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, layout: int = 0, rotaryMode: str = "half"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 query 和 key 执行旋转位置编码 (RoPE) 计算

    Args:
        query: 查询张量，shape 为 (B, S, N, D) 或 (B, N, S, D)
        key: 键张量，shape 同 query
        cos: 余弦位置编码，shape 为 (S, D/2) 或 (B, S, D/2)
        sin: 正弦位置编码，shape 同 cos
        layout: 输入布局 (0: [B,S,N,D], 1: [B,N,S,D])
        rotaryMode: 旋转模式 ("half": 连续半分式，"interleaved": 交错式)

    Returns:
        query_out: 旋转后的查询张量
        key_out: 旋转后的键张量

    Examples:
        >>> B, S, N, D = 2, 4, 8, 128
        >>> query = torch.randn(B, S, N, D)
        >>> key = torch.randn(B, S, N, D)
        >>> cos = torch.randn(S, D // 2)
        >>> sin = torch.randn(S, D // 2)
        >>> q_out, k_out = apply_rotary_pos_emb(query, key, cos, sin)
    """
    # 检测输入 dtype
    input_dtype = query.dtype

    # FP16/BF16 输入需要升到 FP32 计算以保证精度
    # FP32/FP64 输入保持原样计算
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    # 转换到计算精度
    query_compute = query.to(compute_dtype)
    key_compute = key.to(compute_dtype)
    cos_compute = cos.to(compute_dtype)
    sin_compute = sin.to(compute_dtype)

    def rotate_half(x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        旋转输入张量的一半维度

        Args:
            x: 输入张量
            mode: 旋转模式

        Returns:
            旋转后的张量
        """
        if mode == "interleaved":
            # GPT-J 风格的交错式旋转
            x1 = x[..., ::2]  # 取偶数索引
            x2 = x[..., 1::2]  # 取奇数索引
            rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        else:
            # LLaMA/Meta 风格的连续半分式旋转
            half_dim = x.shape[-1] // 2
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            rotated = torch.cat([-x2, x1], dim=-1)
        return rotated

    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: str) -> torch.Tensor:
        """
        对单个张量应用 RoPE

        Args:
            x: 输入张量
            cos: 余弦编码
            sin: 正弦编码
            mode: 旋转模式

        Returns:
            旋转后的张量
        """
        # 调整 cos/sin 的 shape 以匹配输入
        # cos/sin: (S, D/2) 或 (B, S, D/2)
        # 需要扩展到 (B, S, N, D) 或 (B, N, S, D)

        if cos.dim() == 2:
            # cos: (S, D/2) -> 需要扩展到 (B, S, 1, D)
            cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.dim() == 3:
            # cos: (B, S, D/2) -> 需要扩展到 (B, S, 1, D)
            cos = cos.unsqueeze(2)  # (B, S, 1, D/2)
            sin = sin.unsqueeze(2)

        # 如果 layout=1 (B,N,S,D)，需要调整
        if layout == 1:
            cos = cos.transpose(1, 2)  # (B, 1, S, D/2)
            sin = sin.transpose(1, 2)

        # 重复 cos/sin 到完整的 head_dim
        # interleaved 模式需要 cos/sin 也是 interleaved 格式
        if mode == "interleaved":
            # interleaved 格式: [c1, c1, c2, c2, ...]
            cos_full = torch.zeros_like(cos.repeat(1, 1, 1, 2))
            sin_full = torch.zeros_like(sin.repeat(1, 1, 1, 2))
            cos_full[..., ::2] = cos  # 偶数位置
            cos_full[..., 1::2] = cos  # 奇数位置
            sin_full[..., ::2] = sin
            sin_full[..., 1::2] = sin
            cos = cos_full
            sin = sin_full
        else:
            # half 格式: [c1, c2, ..., c1, c2, ...]
            cos = cos.repeat(1, 1, 1, 2)
            sin = sin.repeat(1, 1, 1, 2)

        # 应用 RoPE 公式
        x_rotate = rotate_half(x, mode)
        return (x * cos) + (x_rotate * sin)

    # 对 query 和 key 分别应用 RoPE
    query_out = apply_rotary(query_compute, cos_compute, sin_compute, rotaryMode)
    key_out = apply_rotary(key_compute, cos_compute, sin_compute, rotaryMode)

    # 转回原始 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return query_out.to(input_dtype), key_out.to(input_dtype)
    return query_out, key_out


PRECISION_THRESHOLDS = {
    torch.float16: 2**-10,
    torch.bfloat16: 2**-7,
    torch.float32: 2**-13,
}


def precision_compare(actual: torch.Tensor, golden: torch.Tensor, name: str = "output") -> dict:
    threshold = PRECISION_THRESHOLDS.get(golden.dtype)
    if threshold is None:
        raise ValueError(f"Unsupported precision threshold dtype: {golden.dtype}")

    actual_f = actual.float()
    golden_f = golden.float()
    rel_err = (actual_f - golden_f).abs() / (golden_f.abs() + 1e-7)
    mere = rel_err.mean().item()
    mare = rel_err.max().item()
    max_abs_err = (actual_f - golden_f).abs().max().item()
    passed = mere < threshold and mare < 10 * threshold
    return {
        "name": name,
        "dtype": str(golden.dtype),
        "threshold": threshold,
        "mere": mere,
        "mare": mare,
        "max_abs_err": max_abs_err,
        "passed": passed,
    }


def print_precision_result(result: dict):
    print(
        f"{result['name']} precision: "
        f"MERE={result['mere']:.6e} "
        f"MARE={result['mare']:.6e} "
        f"threshold={result['threshold']:.6e} "
        f"max_abs_err={result['max_abs_err']:.6e} "
        f"passed={result['passed']}"
    )
    if_pass = result["passed"]
    return if_pass


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    # tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    # tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

UB_LIMIT = 196608

DTYPE_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}


@tilelang.jit(pass_configs=PASS_CONFIGS)
def apply_rotary_pos_emb_tile_kernel(
    B,
    S,
    N,
    D,
    layout,
    mode,
    dtype="float16",
    Block_M=8,
    S_TILE=1,
    D_TILE=64,
    num_stages=2,
):
    # layout0:[B,S,N,D]  layout1:[B,N,S,D]
    HalfD = D // 2
    compute_dtype = "float32"
    need_cast = dtype != compute_dtype
    # elem_bytes = 4 if dtype == "float32" else 2
    elem_bytes = 4
    BN_total = B * N
    m_num = (BN_total + Block_M - 1) // Block_M
    s_num = (S + S_TILE - 1) // S_TILE
    # d_num = (HalfD + D_TILE - 1) // D_TILE
    D_TILE = D
    d_num = 1
    tile_num = s_num * d_num
    Dim1 = N if layout == 1 else S
    Dim2 = S if layout == 1 else N
    print(f"apply_rotary_pos_emb_tile_kernel: B={B}, S={S}, N={N}, D={D}, layout={layout}, mode={mode}, dtype={dtype}")

    @T.prim_func
    def kernel(
        x_in_q: T.Tensor([B, Dim1, Dim2, D], dtype),
        x_in_k: T.Tensor([B, Dim1, Dim2, D], dtype),
        cos_full: T.Tensor([S, D], "float32"),
        sin_full: T.Tensor([S, D], "float32"),
        # cos_full: T.Tensor([S, D], dtype),
        # sin_full: T.Tensor([S, D], dtype),
        x_out_q: T.Tensor([B, Dim1, Dim2, D], dtype),
        x_out_k: T.Tensor([B, Dim1, Dim2, D], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):  # noqa: SIM117
            with T.Scope("V"):
                # ---- 双缓冲 buffer (手动 MTE2→V→MTE3 三阶段流水) ----
                cos_ub = T.alloc_ub((2, S_TILE, D_TILE), "float32")
                sin_ub = T.alloc_ub((2, S_TILE, D_TILE), "float32")
                sin_load = T.alloc_ub((2, S_TILE, D_TILE), "float32")
                x_tile = T.alloc_ub((2, S_TILE, D_TILE), dtype)
                out_tile = T.alloc_ub((2, S_TILE, D_TILE), dtype)
                # V 阶段计算 buffer (单份即可, V 内部串行)
                x_fp32 = T.alloc_ub((S_TILE, D_TILE), "float32")
                x_rotate = T.alloc_ub((S_TILE, D_TILE), "float32")
                out_fp32 = T.alloc_ub((S_TILE, D_TILE), "float32")

                # 构造 sin_sign: half=[-1..-1, 1..1], interleaved=[-1,1,-1,1..]
                sin_sign = T.alloc_ub((D_TILE,), "float32")
                if mode == "half":
                    T.tile.fill(sin_sign, 1.0)
                    for i in T.serial(HalfD):
                        sin_sign[i] = -1.0
                else:
                    T.tile.fill(sin_sign, -1.0)
                    for i in T.unroll(HalfD):
                        sin_sign[2 * i + 1] = 1.0

                # ---- 构造 gather rotate_mask (保留原 kernel 方案) ----
                idx_i32 = T.alloc_ub((S_TILE, D_TILE), "int32")
                idx_i16 = T.alloc_ub((S_TILE, D_TILE), "int16")
                half_i16 = T.alloc_ub((S_TILE, D_TILE), "int16")
                mask_i16 = T.alloc_ub((S_TILE, D_TILE), "int16")
                mask_f32 = T.alloc_ub((S_TILE, D_TILE), "float32")
                mask_i32 = T.alloc_ub((S_TILE, D_TILE), "int32")
                rotate_mask_ub = T.alloc_ub((S_TILE, D_TILE), "uint32")

                T.tile.createvecindex(idx_i32, 0)
                T.copy(idx_i32, idx_i16)

                if mode == "half":
                    T.tile.fill(half_i16, HalfD)
                else:
                    T.tile.fill(half_i16, 1)
                T.tile.bitwise_xor(mask_i16, idx_i16, half_i16)

                T.copy(mask_i16, mask_f32)
                T.copy(mask_f32, mask_i32)

                T.tile.mul(mask_i32, mask_i32, elem_bytes)
                T.reinterpretcast(rotate_mask_ub, mask_i32, "uint32_t")

                # ---- (b,n) 外层循环 ----
                for bn_local in T.serial(Block_M):
                    bn_idx = cid * Block_M + bn_local
                    if bn_idx < BN_total:
                        b_idx = bn_idx // N
                        n_idx = bn_idx % N

                        # ---- 初始化 MTE3→MTE2 flag (允许 buffer 0/1 被 MTE2 写) ----
                        T.set_flag("mte3", "mte2", 0)
                        T.set_flag("mte3", "mte2", 1)
                        T.wait_flag("mte3", "mte2", 0)

                        # ---- preload tile0 到 buffer 0 (纯 MTE2, 无 V 指令) ----
                        s_base_0 = 0
                        s_end_0 = T.min(S_TILE, S)
                        T.copy(cos_full[s_base_0:s_end_0, :], cos_ub[0, :, :])
                        T.copy(sin_full[s_base_0:s_end_0, :], sin_load[0, :, :])
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

                        # ---- 主循环: preload next + compute current + store current ----
                        for tile_id in T.serial(tile_num):
                            cur = tile_id % 2
                            nxt = (tile_id + 1) % 2
                            cur_s_base = tile_id * S_TILE
                            cur_s_end = cur_s_base + T.min(S_TILE, S - cur_s_base)

                            # 1) preload next tile 到 nxt buffer (提前发, 不等当前 compute)
                            if tile_id + 1 < tile_num:
                                T.wait_flag("mte3", "mte2", nxt)
                                next_s_base = (tile_id + 1) * S_TILE
                                next_s_end = next_s_base + T.min(S_TILE, S - next_s_base)
                                T.copy(cos_full[next_s_base:next_s_end, :], cos_ub[nxt, :, :])
                                T.copy(sin_full[next_s_base:next_s_end, :], sin_load[nxt, :, :])
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

                            # 2) compute current tile (V 阶段, 含 sin 预处理)
                            T.wait_flag("mte2", "v", cur)
                            # sin 预处理: sin_ub = sin_load * sin_sign (V 指令, 在 V 阶段做)
                            for s in T.unroll(S_TILE):
                                T.tile.mul(sin_ub[cur, s, :], sin_load[cur, s, :], sin_sign)

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

                            # 3) store current tile (MTE3 阶段)
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

                        # ---- 收尾: 等所有 MTE3 完成 ----
                        T.wait_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 1)

    return kernel


PASS_CONFIGS_FLAT = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}
_FLAT_VEC_NUM = 2
_FLAT_CAST_LOW2HIGH = "CAST_NONE"
_FLAT_CAST_HIGH2LOW = "CAST_RINT"


@tilelang.jit(out_idx=[4, 5], pass_configs=PASS_CONFIGS_FLAT)
def _flat_kernel_half(M, D, block_M, dtype="float16"):
    """Half mode RoPE kernel: split-and-compute (no gather). Q+K fused."""
    D_HALF = D // 2
    sub_block_M = block_M // _FLAT_VEC_NUM
    m_num = T.ceildiv(M, block_M)
    need_cast = dtype in ("float16", "bfloat16")
    cal_dtype = "float32" if need_cast else dtype
    count = sub_block_M * D_HALF

    @T.prim_func
    def main(
        Q: T.Tensor((M, D), dtype),
        K: T.Tensor((M, D), dtype),
        COS: T.Tensor((M, D_HALF), dtype),
        SIN: T.Tensor((M, D_HALF), dtype),
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
                if need_cast:
                    cos_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                    sin_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                    x_first_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                    x_second_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                    y_first_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                    y_second_h = T.alloc_ub((sub_block_M, D_HALF), dtype)
                    T.copy(COS[row_start, 0], cos_h)
                    T.copy(SIN[row_start, 0], sin_h)
                    T.tile.cast(cos_ub, cos_h, _FLAT_CAST_LOW2HIGH, count)
                    T.tile.cast(sin_ub, sin_h, _FLAT_CAST_LOW2HIGH, count)
                    # Q
                    T.copy(Q[row_start, 0], x_first_h)
                    T.copy(Q[row_start, D_HALF], x_second_h)
                    T.tile.cast(x_first, x_first_h, _FLAT_CAST_LOW2HIGH, count)
                    T.tile.cast(x_second, x_second_h, _FLAT_CAST_LOW2HIGH, count)
                    T.tile.mul(y_first, x_second, sin_ub)
                    T.tile.mul(y_first, y_first, -1.0)
                    T.tile.mul_add_dst(y_first, x_first, cos_ub)
                    T.tile.mul(y_second, x_first, sin_ub)
                    T.tile.mul_add_dst(y_second, x_second, cos_ub)
                    T.tile.cast(y_first_h, y_first, _FLAT_CAST_HIGH2LOW, count)
                    T.tile.cast(y_second_h, y_second, _FLAT_CAST_HIGH2LOW, count)
                    T.copy(y_first_h, Q_OUT[row_start, 0])
                    T.copy(y_second_h, Q_OUT[row_start, D_HALF])
                    # K
                    T.copy(K[row_start, 0], x_first_h)
                    T.copy(K[row_start, D_HALF], x_second_h)
                    T.tile.cast(x_first, x_first_h, _FLAT_CAST_LOW2HIGH, count)
                    T.tile.cast(x_second, x_second_h, _FLAT_CAST_LOW2HIGH, count)
                    T.tile.mul(y_first, x_second, sin_ub)
                    T.tile.mul(y_first, y_first, -1.0)
                    T.tile.mul_add_dst(y_first, x_first, cos_ub)
                    T.tile.mul(y_second, x_first, sin_ub)
                    T.tile.mul_add_dst(y_second, x_second, cos_ub)
                    T.tile.cast(y_first_h, y_first, _FLAT_CAST_HIGH2LOW, count)
                    T.tile.cast(y_second_h, y_second, _FLAT_CAST_HIGH2LOW, count)
                    T.copy(y_first_h, K_OUT[row_start, 0])
                    T.copy(y_second_h, K_OUT[row_start, D_HALF])
                else:
                    T.copy(COS[row_start, 0], cos_ub)
                    T.copy(SIN[row_start, 0], sin_ub)
                    # Q
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
                    # K
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


@tilelang.jit(out_idx=[4, 5], pass_configs=PASS_CONFIGS_FLAT)
def _flat_kernel_interleaved(M, D, block_M, dtype="float16"):
    """Interleaved mode RoPE kernel: gather + sign mask. Q+K fused."""
    sub_block_M = block_M // _FLAT_VEC_NUM
    m_num = T.ceildiv(M, block_M)
    need_cast = dtype in ("float16", "bfloat16")
    cal_dtype = "float32" if need_cast else dtype
    count = sub_block_M * D

    @T.prim_func
    def main(
        Q: T.Tensor((M, D), dtype),
        K: T.Tensor((M, D), dtype),
        COS: T.Tensor((M, D), dtype),
        SIN: T.Tensor((M, D), dtype),
        Q_OUT: T.Tensor((M, D), dtype),
        K_OUT: T.Tensor((M, D), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_start = cid * block_M + vid * sub_block_M
            with T.Scope("V"):
                cos_ub = T.alloc_ub((sub_block_M, D), cal_dtype)
                sin_ub = T.alloc_ub((sub_block_M, D), cal_dtype)
                x = T.alloc_ub((sub_block_M, D), cal_dtype)
                x_rotate = T.alloc_ub((sub_block_M, D), cal_dtype)
                out = T.alloc_ub((sub_block_M, D), cal_dtype)
                # gather mask
                idx_ub = T.alloc_ub((sub_block_M, D), "int32")
                T.tile.createvecindex(idx_ub, 0)
                idx_i16 = T.alloc_ub((sub_block_M, D), "int16")
                T.copy(idx_ub, idx_i16)
                ones_ub = T.alloc_ub((sub_block_M, D), "int16")
                T.tile.fill(ones_ub, 1)
                mask_i16 = T.alloc_ub((sub_block_M, D), "int16")
                T.tile.bitwise_xor(mask_i16, idx_i16, ones_ub)
                mask_f32 = T.alloc_ub((sub_block_M, D), "float32")
                T.copy(mask_i16, mask_f32)
                mask_i32 = T.alloc_ub((sub_block_M, D), "int32")
                T.copy(mask_f32, mask_i32)
                T.tile.mul(mask_i32, mask_i32, 4)
                mask_ub = T.alloc_ub((sub_block_M, D), "uint32")
                T.reinterpretcast(mask_ub, mask_i32, "uint32_t")
                if need_cast:
                    cos_h = T.alloc_ub((sub_block_M, D), dtype)
                    sin_h = T.alloc_ub((sub_block_M, D), dtype)
                    x_h = T.alloc_ub((sub_block_M, D), dtype)
                    out_h = T.alloc_ub((sub_block_M, D), dtype)
                    T.copy(COS[row_start, 0], cos_h)
                    T.copy(SIN[row_start, 0], sin_h)
                    T.tile.cast(cos_ub, cos_h, _FLAT_CAST_LOW2HIGH, count)
                    T.tile.cast(sin_ub, sin_h, _FLAT_CAST_LOW2HIGH, count)
                    # Q
                    T.copy(Q[row_start, 0], x_h)
                    T.tile.cast(x, x_h, _FLAT_CAST_LOW2HIGH, count)
                    T.tile.gather(x_rotate, x, mask_ub, 0)
                    T.tile.mul(out, x, cos_ub)
                    T.tile.mul(x_rotate, x_rotate, sin_ub)
                    T.tile.add(out, out, x_rotate)
                    T.tile.cast(out_h, out, _FLAT_CAST_HIGH2LOW, count)
                    T.copy(out_h, Q_OUT[row_start, 0])
                    # K
                    T.copy(K[row_start, 0], x_h)
                    T.tile.cast(x, x_h, _FLAT_CAST_LOW2HIGH, count)
                    T.tile.gather(x_rotate, x, mask_ub, 0)
                    T.tile.mul(out, x, cos_ub)
                    T.tile.mul(x_rotate, x_rotate, sin_ub)
                    T.tile.add(out, out, x_rotate)
                    T.tile.cast(out_h, out, _FLAT_CAST_HIGH2LOW, count)
                    T.copy(out_h, K_OUT[row_start, 0])
                else:
                    T.copy(COS[row_start, 0], cos_ub)
                    T.copy(SIN[row_start, 0], sin_ub)
                    # Q
                    T.copy(Q[row_start, 0], x)
                    T.tile.gather(x_rotate, x, mask_ub, 0)
                    T.tile.mul(out, x, cos_ub)
                    T.tile.mul(x_rotate, x_rotate, sin_ub)
                    T.tile.add(out, out, x_rotate)
                    T.copy(out, Q_OUT[row_start, 0])
                    # K
                    T.copy(K[row_start, 0], x)
                    T.tile.gather(x_rotate, x, mask_ub, 0)
                    T.tile.mul(out, x, cos_ub)
                    T.tile.mul(x_rotate, x_rotate, sin_ub)
                    T.tile.add(out, out, x_rotate)
                    T.copy(out, K_OUT[row_start, 0])

    return main


def _flat_choose_block_M(D, dtype, mode):
    UB_LIMIT = 180000
    need_cast = dtype in ("float16", "bfloat16")
    if mode == "half":
        if need_cast:
            max_sub = max(2, UB_LIMIT // (20 * D))
        else:
            max_sub = max(2, UB_LIMIT // (14 * D))
    else:
        if need_cast:
            max_sub = max(2, UB_LIMIT // (50 * D))
        else:
            max_sub = max(2, UB_LIMIT // (42 * D))
    block_M = min(max_sub, 128) * _FLAT_VEC_NUM
    block_M = max(block_M, 4)
    return block_M


def _flat_expand_cos_sin(cos, sin, M, S, N, B, layout, mode):
    D_half = cos.shape[-1]
    cos_f = cos.to(torch.float32)
    sin_f = sin.to(torch.float32)
    if layout == 0:
        cos_exp = cos_f.unsqueeze(0).unsqueeze(2).expand(B, S, N, D_half).contiguous().view(M, D_half)
        sin_exp = sin_f.unsqueeze(0).unsqueeze(2).expand(B, S, N, D_half).contiguous().view(M, D_half)
    else:
        cos_exp = cos_f.unsqueeze(0).unsqueeze(1).expand(B, N, S, D_half).contiguous().view(M, D_half)
        sin_exp = sin_f.unsqueeze(0).unsqueeze(1).expand(B, N, S, D_half).contiguous().view(M, D_half)
    if mode == "interleaved":
        cos_exp = cos_exp.unsqueeze(-1).expand(M, D_half, 2).contiguous().view(M, D_half * 2)
        sin_exp = torch.stack([-sin_exp, sin_exp], dim=-1).flatten(-1)
    return cos_exp, sin_exp


_flat_kernel_cache = {}


def apply_rotary_pos_emb_flat(query, key, cos, sin, layout=0, rotaryMode="half"):
    """Flat kernel: reshape [M,D] + cos/sin expand + 2D kernel。适用于 BN<20 或 S<=4 的慢 case。"""
    if layout == 0:
        B, S, N, D = query.shape
    else:
        B, N, S, D = query.shape
    M = B * S * N if layout == 0 else B * N * S
    input_dtype = query.dtype
    if input_dtype == torch.float16:
        tl_dtype = "float16"
    elif input_dtype == torch.bfloat16:
        tl_dtype = "bfloat16"
    else:
        tl_dtype = "float32"

    q_2d = query.contiguous().reshape(M, D)
    k_2d = key.contiguous().reshape(M, D)
    cos_exp, sin_exp = _flat_expand_cos_sin(cos, sin, M, S, N, B, layout, rotaryMode)
    cos_exp = cos_exp.to(input_dtype)
    sin_exp = sin_exp.to(input_dtype)

    block_M = _flat_choose_block_M(D, tl_dtype, rotaryMode)
    cache_key = (M, D, block_M, tl_dtype, rotaryMode)
    if cache_key not in _flat_kernel_cache:
        if rotaryMode == "half":
            _flat_kernel_cache[cache_key] = _flat_kernel_half(M, D, block_M, dtype=tl_dtype)
        else:
            _flat_kernel_cache[cache_key] = _flat_kernel_interleaved(M, D, block_M, dtype=tl_dtype)
    kernel = _flat_kernel_cache[cache_key]

    q_out_2d, k_out_2d = kernel(q_2d, k_2d, cos_exp, sin_exp)
    q_out = q_out_2d.reshape(query.shape)
    k_out = k_out_2d.reshape(key.shape)
    return q_out, k_out


def apply_rotary_pos_emb_tl(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = 0,
    rotaryMode: str = "half",
):
    if layout == 0:
        B, S, N, D = query.shape
    else:
        B, N, S, D = query.shape

    dtype_str = DTYPE_STR[query.dtype]

    BN_total = B * N
    M_total = B * N * S

    # kernel 选择策略 (基于 cann-bench 实测数据):
    # flat 赢: BN<20 (单核瓶颈), S<=31 (tile 流水不足), M<=50000 (小数据 flat 开销低)
    # tile 赢: 大 BN + 大 S (S-tile 流水 + 手动 flag 三阶段流水优势)
    use_flat = (BN_total < 20) or (S <= 31) or (M_total <= 50000)

    if use_flat:
        return apply_rotary_pos_emb_flat(query, key, cos, sin, layout, rotaryMode)
    else:
        cos_fp32 = cos.to(torch.float32)
        sin_fp32 = sin.to(torch.float32)

        if rotaryMode == "half":
            cos_full = cos_fp32.repeat(1, 2)
            sin_full = sin_fp32.repeat(1, 2)
        else:
            cos_full = cos_fp32.repeat_interleave(2, dim=-1)
            sin_full = sin_fp32.repeat_interleave(2, dim=-1)

        UB_LIMIT = 180224
        ds = 4 if dtype_str == "float32" else 2
        s_tile_max = (UB_LIMIT - 4 * D) // (D * (64 + 2 * ds))
        S_TILE = min(max(s_tile_max, 1), S)

        if BN_total < 20:
            Block_M = 1
        else:
            Block_M = 3
        num_stages = 2
        D_TILE = D

        kernel = apply_rotary_pos_emb_tile_kernel(
            B,
            S,
            N,
            D,
            layout,
            rotaryMode,
            dtype=dtype_str,
            Block_M=Block_M,
            S_TILE=S_TILE,
            D_TILE=D_TILE,
            num_stages=num_stages,
        )

        q_out = torch.empty_like(query)
        k_out = torch.empty_like(key)

        kernel(query, key, cos_full, sin_full, q_out, k_out)
        return q_out, k_out


def main():
    device = torch.device("npu")
    dtype = torch.float32

    rotaryMode = "interleaved"  # 'interleaved'  "half"
    # B, S, N, D = 16, 61, 16, 64   # 16, 512, 16, 128
    # B, S, N, D = 7, 1009, 7, 128   # 16, 512, 16, 128

    # B, S, N, D = 8, 2048, 16, 128   # case1
    # B, S, N, D = 7, 102, 31, 64   # case2
    B, S, N, D = 7, 1009, 7, 128
    layout = 0
    shape = (B, S, N, D)

    # layout = 1
    # shape = (B, N, S, D)
    generator = torch.Generator().manual_seed(42)

    query = torch.randn(shape, dtype=dtype, generator=generator).to(device)
    key = torch.randn(shape, dtype=dtype, generator=generator).to(device)
    cos = torch.randn((S, D // 2), dtype=dtype, generator=generator).to(device)
    sin = torch.randn((S, D // 2), dtype=dtype, generator=generator).to(device)

    dtype_str = DTYPE_STR[query.dtype]

    # Block_M = B * N // 20
    if B * N < 20:
        Block_M = 1
    else:
        Block_M = (B * N + 20 - 1) // 20
    # 动态 S_TILE: 根据 D/dtype 计算 UB 预算内最大值 (不要求整除 S, 尾块用 min 处理)
    UB_LIMIT = 180224  # 192KB - 16KB 余量
    ds = 4 if dtype_str == "float32" else 2
    s_tile_max = (UB_LIMIT - 4 * D) // (D * (64 + 2 * ds))
    S_TILE = min(s_tile_max, S)
    if S_TILE < 1:
        S_TILE = 1
    num_stages = 2
    D_TILE = D

    cos_fp32 = cos.to(torch.float32)
    sin_fp32 = sin.to(torch.float32)
    if rotaryMode == "half":
        cos_full = cos_fp32.repeat(1, 2)
        sin_full = sin_fp32.repeat(1, 2)
    else:
        cos_full = cos_fp32.repeat_interleave(2, dim=-1)
        sin_full = sin_fp32.repeat_interleave(2, dim=-1)

    kernel = apply_rotary_pos_emb_tile_kernel(
        B,
        S,
        N,
        D,
        layout,
        rotaryMode,
        dtype=dtype_str,
        Block_M=Block_M,
        S_TILE=S_TILE,
        D_TILE=D_TILE,
        num_stages=num_stages,
    )

    q_out, k_out = torch.empty_like(query), torch.empty_like(key)
    kernel(query, key, cos_full, sin_full, q_out, k_out)

    q_ref, k_ref = apply_rotary_pos_emb(query, key, cos, sin, layout, rotaryMode)
    torch.npu.synchronize()

    print("q_out shape:", tuple(q_out.shape))
    print("k_out shape:", tuple(k_out.shape))
    if_pass_q = print_precision_result(precision_compare(q_out, q_ref, "query"))
    if_pass_k = print_precision_result(precision_compare(k_out, k_ref, "key"))

    # def run_tilelang():
    #     kernel(query, key, cos_full, sin_full, q_out, k_out)

    def run_torch():
        apply_rotary_pos_emb(query, key, cos, sin, layout, rotaryMode)

    # tile_ms = bench_ms(run_tilelang, warmup=10, repeat=100)
    torch_ms = bench_us(run_torch, warmup=10, repeat=100)

    # print(f"tilelang: {tile_ms:.6f} ms")
    print(f"torch baseline: {torch_ms:.6f} us")
    # print(f"speedup: {torch_ms / tile_ms:.3f}x")
    if if_pass_q and if_pass_k:
        print("Kernel Output Match!")


if __name__ == "__main__":
    main()
