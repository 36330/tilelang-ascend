import tilelang
import tilelang.language as T
import torch

RSQRT_2 = 0.7071067811865476
SQRT_8_PI = 1.5957691216057307
COEFF_044715 = 0.044715
COEFF_044715_SQRT_8_PI = COEFF_044715 * SQRT_8_PI

ERF_P = 0.3275911
ERF_A1 = 0.254829592
ERF_A2 = -0.284496736
ERF_A3 = 1.421413741
ERF_A4 = -1.453152027
ERF_A5 = 1.061405429

_CAST_LOW2HIGH = "CAST_NONE"
_CAST_HIGH2LOW = "CAST_RINT"

BLOCK_N = 8192
VEC_NUM = 2
_KERNEL_CACHE = {}

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
TORCH_TO_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}

PRECISION_THRESHOLDS = {
    "float16": 2 ** (-10),
    "float32": 2 ** (-13),
    "bfloat16": 2 ** (-7),
}

pipe_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pipe_configs)
def _gelu_tanh_kernel(N, block_N, dtype="float32"):
    tile_elem = block_N // VEC_NUM
    total_tiles = (N + block_N - 1) // block_N
    tiles_per_block = max(2, min(8, (total_tiles + 23) // 24))
    num_blocks = (total_tiles + tiles_per_block - 1) // tiles_per_block

    @T.prim_func
    def main(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((2, tile_elem), dtype)
            y_ub = T.alloc_ub((2, tile_elem), dtype)
            t0 = T.alloc_ub((tile_elem,), dtype)
            base = cid * tiles_per_block * block_N + vid * tile_elem

            with T.Scope("V"):
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)
                T.wait_flag("mte3", "mte2", 0)
                T.copy(X[base], x_ub[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.unroll(0, tiles_per_block - 1):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    cur_off = base + tile * block_N
                    next_off = base + (tile + 1) * block_N

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(X[next_off], x_ub[nxt, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], x_ub[cur, :])
                    T.tile.mul(t0, x_ub[cur, :], y_ub[cur, :])
                    T.tile.mul(t0, t0, COEFF_044715_SQRT_8_PI)
                    T.tile.axpy(t0, x_ub[cur, :], SQRT_8_PI)
                    T.tile.sigmoid(y_ub[cur, :], t0)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], y_ub[cur, :])
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(y_ub[cur, :], Y[cur_off])
                    T.set_flag("mte3", "mte2", cur)

                last_stage = (tiles_per_block - 1) % 2
                last_off = base + (tiles_per_block - 1) * block_N
                T.wait_flag("mte2", "v", last_stage)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], x_ub[last_stage, :])
                T.tile.mul(t0, x_ub[last_stage, :], y_ub[last_stage, :])
                T.tile.mul(t0, t0, COEFF_044715_SQRT_8_PI)
                T.tile.axpy(t0, x_ub[last_stage, :], SQRT_8_PI)
                T.tile.sigmoid(y_ub[last_stage, :], t0)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], y_ub[last_stage, :])
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(y_ub[last_stage, :], Y[last_off])
                T.set_flag("mte3", "mte2", last_stage)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pipe_configs)
def _gelu_tanh_cast_kernel(N, block_N, in_dtype="float16", cal_dtype="float32"):
    tile_elem = block_N // VEC_NUM
    total_tiles = (N + block_N - 1) // block_N
    tiles_per_block = max(2, min(8, (total_tiles + 23) // 24))
    num_blocks = (total_tiles + tiles_per_block - 1) // tiles_per_block

    @T.prim_func
    def main(X: T.Tensor((N,), in_dtype), Y: T.Tensor((N,), in_dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((2, tile_elem), cal_dtype)
            y_ub = T.alloc_ub((2, tile_elem), cal_dtype)
            t0 = T.alloc_ub((tile_elem,), cal_dtype)
            x_h = T.alloc_ub((2, tile_elem), in_dtype)
            y_h = T.alloc_ub((2, tile_elem), in_dtype)
            base = cid * tiles_per_block * block_N + vid * tile_elem

            with T.Scope("V"):
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)
                T.wait_flag("mte3", "mte2", 0)
                T.copy(X[base], x_h[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.unroll(0, tiles_per_block - 1):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    cur_off = base + tile * block_N
                    next_off = base + (tile + 1) * block_N

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(X[next_off], x_h[nxt, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.cast(x_ub[cur, :], x_h[cur, :], _CAST_LOW2HIGH, tile_elem)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], x_ub[cur, :])
                    T.tile.mul(t0, x_ub[cur, :], y_ub[cur, :])
                    T.tile.mul(t0, t0, COEFF_044715_SQRT_8_PI)
                    T.tile.axpy(t0, x_ub[cur, :], SQRT_8_PI)
                    T.tile.sigmoid(y_ub[cur, :], t0)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], y_ub[cur, :])
                    T.tile.cast(y_h[cur, :], y_ub[cur, :], _CAST_HIGH2LOW, tile_elem)
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(y_h[cur, :], Y[cur_off])
                    T.set_flag("mte3", "mte2", cur)

                last_stage = (tiles_per_block - 1) % 2
                last_off = base + (tiles_per_block - 1) * block_N
                T.wait_flag("mte2", "v", last_stage)
                T.tile.cast(x_ub[last_stage, :], x_h[last_stage, :], _CAST_LOW2HIGH, tile_elem)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], x_ub[last_stage, :])
                T.tile.mul(t0, x_ub[last_stage, :], y_ub[last_stage, :])
                T.tile.mul(t0, t0, COEFF_044715_SQRT_8_PI)
                T.tile.axpy(t0, x_ub[last_stage, :], SQRT_8_PI)
                T.tile.sigmoid(y_ub[last_stage, :], t0)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], y_ub[last_stage, :])
                T.tile.cast(y_h[last_stage, :], y_ub[last_stage, :], _CAST_HIGH2LOW, tile_elem)
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(y_h[last_stage, :], Y[last_off])
                T.set_flag("mte3", "mte2", last_stage)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pipe_configs)
def _gelu_exact_hw_kernel(N, block_N, in_dtype="float16", cal_dtype="float32"):
    tile_elem = block_N // VEC_NUM
    total_tiles = (N + block_N - 1) // block_N
    tiles_per_block = max(2, min(8, (total_tiles + 23) // 24))
    num_blocks = (total_tiles + tiles_per_block - 1) // tiles_per_block

    @T.prim_func
    def main(X: T.Tensor((N,), in_dtype), Y: T.Tensor((N,), in_dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((2, tile_elem), cal_dtype)
            y_ub = T.alloc_ub((2, tile_elem), cal_dtype)
            t0 = T.alloc_ub((tile_elem,), cal_dtype)
            z_ub = T.alloc_ub((tile_elem,), cal_dtype)
            x_h = T.alloc_ub((2, tile_elem), in_dtype)
            y_h = T.alloc_ub((2, tile_elem), in_dtype)
            base = cid * tiles_per_block * block_N + vid * tile_elem

            with T.Scope("V"):
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)
                T.wait_flag("mte3", "mte2", 0)
                T.copy(X[base], x_h[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.unroll(0, tiles_per_block - 1):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    cur_off = base + tile * block_N
                    next_off = base + (tile + 1) * block_N

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(X[next_off], x_h[nxt, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.cast(x_ub[cur, :], x_h[cur, :], _CAST_LOW2HIGH, tile_elem)
                    T.tile.mul(z_ub, x_ub[cur, :], RSQRT_2)
                    T.tile.erf(t0, z_ub)
                    T.tile.add(t0, t0, 1.0)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], t0)
                    T.tile.mul(y_ub[cur, :], y_ub[cur, :], 0.5)
                    T.tile.cast(y_h[cur, :], y_ub[cur, :], _CAST_HIGH2LOW, tile_elem)
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(y_h[cur, :], Y[cur_off])
                    T.set_flag("mte3", "mte2", cur)

                last_stage = (tiles_per_block - 1) % 2
                last_off = base + (tiles_per_block - 1) * block_N
                T.wait_flag("mte2", "v", last_stage)
                T.tile.cast(x_ub[last_stage, :], x_h[last_stage, :], _CAST_LOW2HIGH, tile_elem)
                T.tile.mul(z_ub, x_ub[last_stage, :], RSQRT_2)
                T.tile.erf(t0, z_ub)
                T.tile.add(t0, t0, 1.0)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], t0)
                T.tile.mul(y_ub[last_stage, :], y_ub[last_stage, :], 0.5)
                T.tile.cast(y_h[last_stage, :], y_ub[last_stage, :], _CAST_HIGH2LOW, tile_elem)
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(y_h[last_stage, :], Y[last_off])
                T.set_flag("mte3", "mte2", last_stage)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pipe_configs)
def _gelu_exact_as_kernel(N, block_N, dtype="float32"):
    tile_elem = block_N // VEC_NUM
    total_tiles = (N + block_N - 1) // block_N
    tiles_per_block = max(2, min(8, (total_tiles + 23) // 24))
    num_blocks = (total_tiles + tiles_per_block - 1) // tiles_per_block

    @T.prim_func
    def main(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((2, tile_elem), dtype)
            y_ub = T.alloc_ub((2, tile_elem), dtype)
            t0 = T.alloc_ub((tile_elem,), dtype)
            t1 = T.alloc_ub((tile_elem,), dtype)
            t2 = T.alloc_ub((tile_elem,), dtype)
            z_ub = T.alloc_ub((tile_elem,), dtype)
            h0 = T.alloc_ub((tile_elem,), dtype)
            h1 = T.alloc_ub((tile_elem,), dtype)
            ones = T.alloc_ub((tile_elem,), dtype)
            base = cid * tiles_per_block * block_N + vid * tile_elem

            with T.Scope("V"):
                T.tile.fill(ones, 1.0)

                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)
                T.wait_flag("mte3", "mte2", 0)
                T.copy(X[base], x_ub[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.unroll(0, tiles_per_block - 1):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    cur_off = base + tile * block_N
                    next_off = base + (tile + 1) * block_N

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(X[next_off], x_ub[nxt, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.mul(z_ub, x_ub[cur, :], RSQRT_2)
                    T.tile.mul(t0, z_ub, z_ub)
                    T.tile.abs(t1, z_ub)
                    T.tile.mul(t0, t0, -1.0)
                    T.tile.exp(t0, t0)
                    T.tile.mul(t2, t1, ERF_P)
                    T.tile.add(t2, t2, 1.0)
                    T.tile.div(t2, ones, t2)
                    T.tile.fill(h0, ERF_A5)
                    T.tile.fill(h1, ERF_A4)
                    T.tile.mul_add_dst(h1, t2, h0)
                    T.tile.fill(h0, ERF_A3)
                    T.tile.mul_add_dst(h0, t2, h1)
                    T.tile.fill(h1, ERF_A2)
                    T.tile.mul_add_dst(h1, t2, h0)
                    T.tile.fill(h0, ERF_A1)
                    T.tile.mul_add_dst(h0, t2, h1)
                    T.tile.mul(t1, h0, t2)
                    T.tile.mul(t1, t1, t0)
                    T.tile.sub(y_ub[cur, :], ones, t1)
                    T.tile.mul(t0, y_ub[cur, :], -1.0)
                    T.tile.compare(t1, z_ub, 0.0, "GE")
                    T.tile.select(y_ub[cur, :], t1, y_ub[cur, :], t0, "VSEL_TENSOR_TENSOR_MODE")
                    T.tile.add(y_ub[cur, :], y_ub[cur, :], 1.0)
                    T.tile.mul(y_ub[cur, :], y_ub[cur, :], 0.5)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], y_ub[cur, :])
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(y_ub[cur, :], Y[cur_off])
                    T.set_flag("mte3", "mte2", cur)

                last_stage = (tiles_per_block - 1) % 2
                last_off = base + (tiles_per_block - 1) * block_N
                T.wait_flag("mte2", "v", last_stage)
                T.tile.mul(z_ub, x_ub[last_stage, :], RSQRT_2)
                T.tile.mul(t0, z_ub, z_ub)
                T.tile.abs(t1, z_ub)
                T.tile.mul(t0, t0, -1.0)
                T.tile.exp(t0, t0)
                T.tile.mul(t2, t1, ERF_P)
                T.tile.add(t2, t2, 1.0)
                T.tile.div(t2, ones, t2)
                T.tile.fill(h0, ERF_A5)
                T.tile.fill(h1, ERF_A4)
                T.tile.mul_add_dst(h1, t2, h0)
                T.tile.fill(h0, ERF_A3)
                T.tile.mul_add_dst(h0, t2, h1)
                T.tile.fill(h1, ERF_A2)
                T.tile.mul_add_dst(h1, t2, h0)
                T.tile.fill(h0, ERF_A1)
                T.tile.mul_add_dst(h0, t2, h1)
                T.tile.mul(t1, h0, t2)
                T.tile.mul(t1, t1, t0)
                T.tile.sub(y_ub[last_stage, :], ones, t1)
                T.tile.mul(t0, y_ub[last_stage, :], -1.0)
                T.tile.compare(t1, z_ub, 0.0, "GE")
                T.tile.select(y_ub[last_stage, :], t1, y_ub[last_stage, :], t0, "VSEL_TENSOR_TENSOR_MODE")
                T.tile.add(y_ub[last_stage, :], y_ub[last_stage, :], 1.0)
                T.tile.mul(y_ub[last_stage, :], y_ub[last_stage, :], 0.5)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], y_ub[last_stage, :])
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(y_ub[last_stage, :], Y[last_off])
                T.set_flag("mte3", "mte2", last_stage)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pipe_configs)
def _gelu_exact_as_cast_kernel(N, block_N, in_dtype="bfloat16", cal_dtype="float32"):
    tile_elem = block_N // VEC_NUM
    total_tiles = (N + block_N - 1) // block_N
    tiles_per_block = max(2, min(8, (total_tiles + 23) // 24))
    num_blocks = (total_tiles + tiles_per_block - 1) // tiles_per_block

    @T.prim_func
    def main(X: T.Tensor((N,), in_dtype), Y: T.Tensor((N,), in_dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((2, tile_elem), cal_dtype)
            y_ub = T.alloc_ub((2, tile_elem), cal_dtype)
            x_h = T.alloc_ub((2, tile_elem), in_dtype)
            y_h = T.alloc_ub((2, tile_elem), in_dtype)
            t0 = T.alloc_ub((tile_elem,), cal_dtype)
            t1 = T.alloc_ub((tile_elem,), cal_dtype)
            t2 = T.alloc_ub((tile_elem,), cal_dtype)
            z_ub = T.alloc_ub((tile_elem,), cal_dtype)
            h0 = T.alloc_ub((tile_elem,), cal_dtype)
            h1 = T.alloc_ub((tile_elem,), cal_dtype)
            ones = T.alloc_ub((tile_elem,), cal_dtype)
            base = cid * tiles_per_block * block_N + vid * tile_elem

            with T.Scope("V"):
                T.tile.fill(ones, 1.0)

                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)
                T.wait_flag("mte3", "mte2", 0)
                T.copy(X[base], x_h[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.unroll(0, tiles_per_block - 1):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    cur_off = base + tile * block_N
                    next_off = base + (tile + 1) * block_N

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(X[next_off], x_h[nxt, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.cast(x_ub[cur, :], x_h[cur, :], _CAST_LOW2HIGH, tile_elem)
                    T.tile.mul(z_ub, x_ub[cur, :], RSQRT_2)
                    T.tile.mul(t0, z_ub, z_ub)
                    T.tile.abs(t1, z_ub)
                    T.tile.mul(t0, t0, -1.0)
                    T.tile.exp(t0, t0)
                    T.tile.mul(t2, t1, ERF_P)
                    T.tile.add(t2, t2, 1.0)
                    T.tile.div(t2, ones, t2)
                    T.tile.fill(h0, ERF_A5)
                    T.tile.fill(h1, ERF_A4)
                    T.tile.mul_add_dst(h1, t2, h0)
                    T.tile.fill(h0, ERF_A3)
                    T.tile.mul_add_dst(h0, t2, h1)
                    T.tile.fill(h1, ERF_A2)
                    T.tile.mul_add_dst(h1, t2, h0)
                    T.tile.fill(h0, ERF_A1)
                    T.tile.mul_add_dst(h0, t2, h1)
                    T.tile.mul(t1, h0, t2)
                    T.tile.mul(t1, t1, t0)
                    T.tile.sub(y_ub[cur, :], ones, t1)
                    T.tile.mul(t0, y_ub[cur, :], -1.0)
                    T.tile.compare(t1, z_ub, 0.0, "GE")
                    T.tile.select(y_ub[cur, :], t1, y_ub[cur, :], t0, "VSEL_TENSOR_TENSOR_MODE")
                    T.tile.add(y_ub[cur, :], y_ub[cur, :], 1.0)
                    T.tile.mul(y_ub[cur, :], y_ub[cur, :], 0.5)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], y_ub[cur, :])
                    T.tile.cast(y_h[cur, :], y_ub[cur, :], _CAST_HIGH2LOW, tile_elem)
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(y_h[cur, :], Y[cur_off])
                    T.set_flag("mte3", "mte2", cur)

                last_stage = (tiles_per_block - 1) % 2
                last_off = base + (tiles_per_block - 1) * block_N
                T.wait_flag("mte2", "v", last_stage)
                T.tile.cast(x_ub[last_stage, :], x_h[last_stage, :], _CAST_LOW2HIGH, tile_elem)
                T.tile.mul(z_ub, x_ub[last_stage, :], RSQRT_2)
                T.tile.mul(t0, z_ub, z_ub)
                T.tile.abs(t1, z_ub)
                T.tile.mul(t0, t0, -1.0)
                T.tile.exp(t0, t0)
                T.tile.mul(t2, t1, ERF_P)
                T.tile.add(t2, t2, 1.0)
                T.tile.div(t2, ones, t2)
                T.tile.fill(h0, ERF_A5)
                T.tile.fill(h1, ERF_A4)
                T.tile.mul_add_dst(h1, t2, h0)
                T.tile.fill(h0, ERF_A3)
                T.tile.mul_add_dst(h0, t2, h1)
                T.tile.fill(h1, ERF_A2)
                T.tile.mul_add_dst(h1, t2, h0)
                T.tile.fill(h0, ERF_A1)
                T.tile.mul_add_dst(h0, t2, h1)
                T.tile.mul(t1, h0, t2)
                T.tile.mul(t1, t1, t0)
                T.tile.sub(y_ub[last_stage, :], ones, t1)
                T.tile.mul(t0, y_ub[last_stage, :], -1.0)
                T.tile.compare(t1, z_ub, 0.0, "GE")
                T.tile.select(y_ub[last_stage, :], t1, y_ub[last_stage, :], t0, "VSEL_TENSOR_TENSOR_MODE")
                T.tile.add(y_ub[last_stage, :], y_ub[last_stage, :], 1.0)
                T.tile.mul(y_ub[last_stage, :], y_ub[last_stage, :], 0.5)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], y_ub[last_stage, :])
                T.tile.cast(y_h[last_stage, :], y_ub[last_stage, :], _CAST_HIGH2LOW, tile_elem)
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(y_h[last_stage, :], Y[last_off])
                T.set_flag("mte3", "mte2", last_stage)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


def compute_mere_mare(actual, golden):
    actual = actual.float()
    golden = golden.float()
    diff = (actual - golden).abs()
    denom = golden.abs() + 1e-7
    relative_err = diff / denom
    mere = relative_err.mean().item()
    mare = relative_err.max().item()
    return mere, mare


def check_precision(actual, golden, threshold):
    mere, mare = compute_mere_mare(actual, golden)
    passed = (mere < threshold) and (mare < 10 * threshold)
    return passed, mere, mare


def gen_input(shape, dtype_str, value_range):
    torch_dtype = TORCH_DTYPE_MAP[dtype_str]
    low, high = value_range
    x = torch.empty(shape, dtype=torch.float32).uniform_(low, high).to(torch_dtype)
    return x


def run_gelu_kernel(x, approximate="none", block_N=BLOCK_N):
    if not x.is_npu:
        x = x.npu()
    orig_dtype = x.dtype
    N = x.numel()
    x_flat = x.contiguous().view(-1)

    if orig_dtype == torch.float16:
        cache_key = (N, block_N, "fp16_cast_tanh")
        if cache_key not in _KERNEL_CACHE:
            _KERNEL_CACHE[cache_key] = _gelu_tanh_cast_kernel(N, block_N, in_dtype="float16", cal_dtype="float32")
    elif orig_dtype == torch.bfloat16:
        cache_key = (N, block_N, "bf16_cast_tanh")
        if cache_key not in _KERNEL_CACHE:
            _KERNEL_CACHE[cache_key] = _gelu_tanh_cast_kernel(N, block_N, in_dtype="bfloat16", cal_dtype="float32")
    else:
        cache_key = (N, block_N, "fp32_tanh")
        if cache_key not in _KERNEL_CACHE:
            _KERNEL_CACHE[cache_key] = _gelu_tanh_kernel(N, block_N, dtype="float32")

    kernel = _KERNEL_CACHE[cache_key]
    y_flat = kernel(x_flat)
    y = y_flat.view(x.shape)
    return y


def gelu(x, approximate="none"):
    return run_gelu_kernel(x, approximate=approximate)


__all__ = ["gelu"]


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
RTOL_MAP = {"float16": 1e-3, "bfloat16": 1e-2, "float32": 1e-4}
ATOL_MAP = {"float16": 1e-3, "bfloat16": 1e-2, "float32": 1e-5}


def run_gelu(case_id, shape, dtype_str, approximate, value_range):
    torch_dtype = TORCH_DTYPE_MAP[dtype_str]
    lo, hi = value_range
    x = torch.empty(shape, dtype=torch.float32).uniform_(lo, hi).to(torch_dtype).npu()

    y = gelu(x, approximate=approximate)
    ref = torch.nn.functional.gelu(x, approximate="tanh")

    rtol = RTOL_MAP.get(dtype_str, 1e-2)
    atol = ATOL_MAP.get(dtype_str, 1e-2)
    y_c = y.cpu().float()
    ref_c = ref.cpu().float()
    torch.testing.assert_close(y_c, ref_c, rtol=rtol, atol=atol)

    print(f"Case {case_id}: PASSED  (shape={shape}, dtype={dtype_str})")


if __name__ == "__main__":
    torch.manual_seed(42)

    # (case_id, shape, dtype, value_range)
    test_cases = [
        (1, [1024], "float32", [-3, 3]),
        (2, [1024], "float16", [-3, 3]),
        (3, [1024], "bfloat16", [-3, 3]),
        (4, [1024, 1024], "float32", [-3, 3]),
        (5, [1024, 1024], "float16", [-3, 3]),
        (6, [1024, 1024], "bfloat16", [-3, 3]),
        (7, [2048, 2048], "float32", [-3, 3]),
        (8, [2048, 2048], "float16", [-3, 3]),
        (9, [2048, 2048], "bfloat16", [-3, 3]),
        (10, [363, 367, 373], "float32", [-3, 3]),
        (11, [363, 367, 373], "float16", [-3, 3]),
        (12, [363, 367, 373], "bfloat16", [-3, 3]),
        (13, [4096, 4096], "float32", [-1, 1]),
        (14, [4096, 4096], "float16", [-1, 1]),
        (15, [4096, 4096], "bfloat16", [-1, 1]),
        (16, [8192], "float32", [-3, 3]),
        (17, [8192], "float16", [-3, 3]),
        (18, [8192], "bfloat16", [-3, 3]),
        (19, [1], "float32", [-3, 3]),
        (20, [1], "float16", [-3, 3]),
        (21, [1], "bfloat16", [-3, 3]),
        (22, [3, 7, 13, 4001], "float32", [-3, 3]),
        (23, [3, 7, 13, 4001], "float16", [-3, 3]),
        (24, [3, 7, 13, 4001], "bfloat16", [-3, 3]),
    ]

    print("=" * 70)
    print("Gelu TileLang-Ascend 测试 (torch.nn.functional.gelu 语义)")
    print(f"共 {len(test_cases)} 个测试用例")
    print("=" * 70)

    passed = 0
    failed = 0
    for case_id, shape, dtype, value_range in test_cases:
        try:
            run_gelu(case_id, shape, dtype, "tanh", value_range)
            passed += 1
        except Exception as e:
            print(f"Case {case_id}: FAILED - {e}")
            failed += 1

    print("=" * 70)
    print(f"测试完成: {passed} passed, {failed} failed")
    if failed == 0:
        print("Test Passed!")
