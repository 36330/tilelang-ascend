import torch

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def vec_square(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b_ub = T.alloc_ub((block_M // vec_num, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // vec_num, by * block_N], a_ub)
            T.barrier_all()
            T.square(b_ub, a_ub)
            T.barrier_all()
            T.copy(b_ub, B[bx * block_M + vid * block_M // vec_num, by * block_N])

    return main


def compile_square(M=256, N=256, block_M=128, block_N=256, dtype="float"):
    func = vec_square(M, N, block_M, block_N, dtype)
    return tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target="ascendc")


def test_square_codegen():
    func = compile_square()
    source = func.get_kernel_source()
    assert "AscendC::Mul(" in source
    assert "tl.ascend_square" not in source


def test_square_correctness():
    M, N = 256, 256
    func = compile_square(M, N)

    torch.manual_seed(0)
    a = torch.randn(M, N, dtype=torch.float32).npu()
    torch.npu.synchronize()

    b = func(a)
    ref_b = a * a
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    test_square_codegen()
    test_square_correctness()
    print("Square op tests passed")
