

import torch

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

M = 1024
N = 1024

@tilelang.jit(out_idx=[-1])
def vec_abs(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.barrier_all()
            
            # T.tile.abs(b_ub, a_ub)
            T.abs(b_ub,a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main

func = vec_abs(M,N,128,256)

torch.manual_seed(0)

a = torch.randn(M,N).npu()
torch.npu.synchronize()
print("Init success")

b = func(a)

ref_b = torch.abs(a)

torch.testing.assert_close(b,ref_b,rtol=1e-2,atol=1e-2)
print("Kernel output Match")

