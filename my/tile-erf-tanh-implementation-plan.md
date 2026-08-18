# T.tile.erf / T.tile.tanh Ascend 接入实施方案

## 0. 目标与推荐链路

当前目标是让：

```text
T.tile.erf(dst_ub, src_ub)
T.tile.tanh(dst_ub, src_ub)
```

在 `target="ascendc"` 下直接生成 AscendC 原生高级数学接口：

```text
tilelang/language/ascend_tile.py
  -> tl.ascend_erf / tl.ascend_tanh
  -> InjectTmpBuffer 插入 tmp_ub
  -> src/target/codegen_ascend.cc
  -> AscendC::Erf / AscendC::Tanh
```

不要按普通 `Exp/Abs` 的无 tmp 一元算子接。

原因：

```text
AscendC::Erf / AscendC::Tanh 文档明确说明内部复杂数学计算需要额外临时空间。
接口有两类调用：
  1. AscendC::Erf/Tanh(dst, src, count)
  2. AscendC::Erf/Tanh(dst, src, sharedTmpBuffer, count)

TileLang 当前已有 InjectTmpBuffer 和 _call_intrin_with_optional_tmp 机制。
因此框架侧推荐接成显式 tmp 版本，和 sin/cos/pow/sigmoid 一样由 TileLang 管理 UB workspace。
```

支持边界：

```text
第一版只承诺 ascendc 后端。
dtype 只支持 half / float，对应 TileLang dtype 通常是 float16 / float。
Atlas A2/A3/350、Atlas 推理系列 AI Core 支持；Atlas 200I/500 A2 推理产品和 Vector Core 不支持。
PTO 后端如果没有 TERF / TTANH 宏，先不要标为支持。
```

## 1. 第一阶段：AscendC 后端接 T.tile.erf / T.tile.tanh

### 1.1 修改 Python 前端

文件：

```text
tilelang/language/ascend_tile.py
```

修改位置：放在现有 `exp` / `sigmoid` / `silu` 附近。建议放在 `exp` 后面、`sigmoid` 前面。

新增一个带 tmp 的一元高级数学 helper：

```python
def advanced_unary_op(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    op: str,
    *,
    tmp: Buffer | BufferRegion | None = None,
):
    if isinstance(dst, BufferRegion):
        dst_ptr, dst_extent = _handle_buffer_region(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")
        dst_extent = dst.shape
    if isinstance(src0, BufferRegion):
        src0_ptr, src0_extent = _handle_buffer_region(src0, "r")
    else:
        src0_ptr = src0.access_ptr("r")
        src0_extent = src0.shape

    size_0 = math.prod(dst_extent)
    size_1 = math.prod(src0_extent)
    assert size_0 == size_1, "size must be same"

    return _call_intrin_with_optional_tmp(op, [dst_ptr, src0_ptr, size_0], 2, tmp)
```

新增 `erf`：

```python
def erf(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    *,
    tmp: Buffer | BufferRegion | None = None,
):
    """Performs element-wise error function: dst = erf(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
        tmp: Optional complete UB scratch storage. Its scalar dtype is
            reinterpreted by lowering and has no semantic meaning.
    """
    return advanced_unary_op(dst, src0, "erf", tmp=tmp)
```

新增 `tanh`：

```python
def tanh(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    *,
    tmp: Buffer | BufferRegion | None = None,
):
    """Performs element-wise hyperbolic tangent: dst = tanh(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
        tmp: Optional complete UB scratch storage. Its scalar dtype is
            reinterpreted by lowering and has no semantic meaning.
    """
    return advanced_unary_op(dst, src0, "tanh", tmp=tmp)
```

原因：

```text
T.tile 是 tilelang/language/__init__.py 里的 ascend_tile 模块别名：
  from . import ascend_tile as tile

所以这里加函数后，用户可以直接写 T.tile.erf / T.tile.tanh。
使用 _call_intrin_with_optional_tmp 后，public API 可以接受 tmp=xxx，也可以交给 InjectTmpBuffer 自动插入 tmp。
```

### 1.2 注册 TIR builtin

文件 1：

```text
src/op/ascend.h
```

修改位置：放在 `ascend_exp()` / `ascend_ln()` / `ascend_abs()` 附近。

新增声明：

```cpp
TVM_DLL const Op &ascend_erf();

TVM_DLL const Op &ascend_tanh();
```

文件 2：

```text
src/op/ascend.cc
```

修改位置：放在 `TIR_DEFINE_TL_BUILTIN(ascend_exp)` 附近。

新增注册：

```cpp
TIR_DEFINE_TL_BUILTIN(ascend_erf)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tanh)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

原因：

```text
最终 TIR call 形态是：
  tl.ascend_erf(dst, src, tmp, count)
  tl.ascend_tanh(dst, src, tmp, count)

tmp 未显式传入时，Python 先生成 3 参数 public call，后续 InjectTmpBuffer 在 index=2 插入 tmp。
注册成 4 参数和 sin/cos 的模式保持一致。
```

### 1.3 修改 workspace 注入配置

文件：

```text
src/transform/common/operation_config.h
```

修改位置 1：`GetOperationConfigs()` 的 Ascend op 表，放在 `tl.ascend_exp` 附近。

新增：

```cpp
{"tl.ascend_erf", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
{"tl.ascend_tanh", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
```

修改位置 2：`GetWorkspaceOpConfigs()`，放在 `ascend_sin()` / `ascend_cos()` 附近。

新增：

```cpp
{tl::ascend_erf().get(), {2, true, false}},
{tl::ascend_tanh().get(), {2, true, false}},
```

原因：

```text
tmp_arg_index=2 表示 tmp 插在 dst、src 后面：
  dst, src, tmp, count

ascendc_supported=true 表示 AscendC 自动注入 workspace。
pto_supported=false 表示 PTO 暂不注入，避免误以为 PTO 已支持。
```

### 1.4 修改 AscendC workspace 大小估算

文件：

```text
src/transform/allocate_tmp_buffer.cc
```

修改位置 1：anonymous namespace 中，放在 `EstimateAscendCBroadcastWorkspaceBytes` 或 reduce workspace helper 附近。

新增：

```cpp
int64_t EstimateAscendCErfWorkspaceBytes(const CallNode *call) {
  const DataType dtype = GetAccessPtrDtype(call->args[1]);
  const int64_t src_bytes = GetAccessPtrBytes(call->args[1]);
  const bool is_half = dtype.is_float() && dtype.bits() == 16;
  const bool is_float = dtype.is_float() && dtype.bits() == 32;
  ICHECK(is_half || is_float)
      << "AscendC Erf only supports float16/float32, got " << dtype;

  const int64_t factor = is_float ? 3 : 8;
  return factor * std::max<int64_t>(src_bytes, 256);
}

int64_t EstimateAscendCTanhWorkspaceBytes(const CallNode *call) {
  const DataType dtype = GetAccessPtrDtype(call->args[1]);
  const int64_t src_bytes = GetAccessPtrBytes(call->args[1]);
  const bool is_half = dtype.is_float() && dtype.bits() == 16;
  const bool is_float = dtype.is_float() && dtype.bits() == 32;
  ICHECK(is_half || is_float)
      << "AscendC Tanh only supports float16/float32, got " << dtype;

  const int64_t factor = is_float ? 1 : 4;
  return factor * std::max<int64_t>(src_bytes, 256);
}
```

来源：

```text
asc-devkit/impl/adv_api/tiling/math/erf_tiling_impl.cpp:
  float factor = 3
  half factor = 8
  maxSize = factor * max(inputSize * typeSize, 256)

asc-devkit/impl/adv_api/tiling/math/tanh_tiling_impl.cpp:
  float factor = 1
  half factor = 4
  maxSize = factor * max(inputSize * typeSize, 256)
```

修改位置 2：`GetAscendCWorkspaceSpec()`。

放在 `sin/cos` 或 `pow` 分支附近，新增：

```cpp
if (call->op.same_as(tl::ascend_erf())) {
  return RequireWorkspace(byte_dtype, EstimateAscendCErfWorkspaceBytes(call));
}
if (call->op.same_as(tl::ascend_tanh())) {
  return RequireWorkspace(byte_dtype, EstimateAscendCTanhWorkspaceBytes(call));
}
```

原因：

```text
Erf/Tanh 高级数学接口需要临时空间。
如果这里不加 workspace policy，InjectTmpBuffer 会找不到策略，或者最终生成无 tmp 版本，风险是 CANN 内部 workspace 预留不受 TileLang 管理。
```

### 1.5 修改 AscendC codegen

文件：

```text
src/target/codegen_ascend.cc
```

修改位置：`CodeGenTileLangAscend::VisitExpr_(const CallNode *op, ...)` 中，放在 `ascend_exp()` 附近。

新增：

```cpp
} else if (op->op.same_as(tl::ascend_erf())) {
  TrigOpCodegen(op, "AscendC::Erf");
} else if (op->op.same_as(tl::ascend_tanh())) {
  TrigOpCodegen(op, "AscendC::Tanh");
```

原因：

```text
TrigOpCodegen 的打印逻辑正好适配：
  args = dst, src, tmp, count
  output = AscendC::Erf(dst, src, tmp, count)
  output = AscendC::Tanh(dst, src, tmp, count)

不要用 UnaryVecOpCodegen 接普通无 tmp 路径。
```

### 1.6 修改 VID reduction 的 tile op allow-list

文件：

```text
src/transform/ascend_vid_reduction.cc
```

修改位置：`IsTileOp()` 里的 `tile_ops` 集合，放在一元 op 列表附近。

把：

```cpp
"tl.ascend_exp", "tl.ascend_ln", "tl.ascend_abs",
"tl.ascend_reciprocal", "tl.ascend_sqrt", "tl.ascend_rsqrt",
```

扩展为：

```cpp
"tl.ascend_exp", "tl.ascend_erf", "tl.ascend_tanh",
"tl.ascend_ln", "tl.ascend_abs",
"tl.ascend_reciprocal", "tl.ascend_sqrt", "tl.ascend_rsqrt",
```

原因：

```text
AscendVidReduction 会在 vid 并行拆 UB 行时修改 tile op 的 count。
Erf/Tanh 的最后一个参数也是 count，不加这里会导致 vid 拆分后仍按原物理 tile 大小计算。
```

### 1.7 修改 tail mask propagation

文件：

```text
src/transform/ascend_tail_mask_propagation.cc
```

修改位置：`TryRewriteVectorOp()`，放在 cast/broadcast propagate-only 分支前面。

新增：

```cpp
// Erf/Tanh use the advanced math API with explicit tmp:
//   dst(0), src(1), tmp(2), count(3)
// They do not expose the same mask/repeat overload as basic unary ops in the
// current integration, so keep the full-tile call but propagate the valid
// rectangle for downstream ops such as reduce.
if (call->op.same_as(ascend_erf()) || call->op.same_as(ascend_tanh())) {
  if (call->args.size() >= 2) {
    PropagateUnaryShape(call->args[0], call->args[1]);
  }
  return Stmt();
}
```

原因：

```text
Erf/Tanh 不适合直接放进 UnaryTag：
  1. 当前 call layout 是 dst, src, tmp, count，不是 dst, src, count。
  2. 文档只列出 count / all-count 版本，未列 mask+repeat overload。

保持 full-tile 计算通常仍正确，因为 GM->UB tail copy 会 pad，UB->GM 只写回 valid region。
但必须传播 tail mask，否则后面接 reduce 时 reduce 会丢失 valid region 信息。
```

如果后续要做更强的 tail 优化，可以新增 `tail_advanced_unary` helper，只使用按行 count 调用，不走 mask/repeat 分支：

```text
validCol == physCol:
  AscendC::Erf/Tanh(dst, src, tmp, validRow * physCol)
else:
  for each valid row:
    AscendC::Erf/Tanh(dst[row], src[row], tmp, validCol)
```

第一版不建议先做这步，避免 tmp 复用和 per-row workspace 生命周期复杂化。

## 2. 第二阶段：PTO 后端策略

第一版建议不接 PTO。

需要明确保留现状：

```text
target="pto" 遇到 tl.ascend_erf / tl.ascend_tanh 应该报 unsupported。
不要在 codegen_ascend_pto.cc 里猜 TERF / TTANH 宏。
```

如果你确认 PTO 侧存在宏：

```text
TERF(dst, src)
TTANH(dst, src)
```

再做这些改动：

文件：

```text
src/target/codegen_ascend_pto.cc
```

在 `VisitExpr_` 的 unary vector ops 分支新增：

```cpp
} else if (op->op.same_as(tl::ascend_erf())) {
  UnaryVecOpCodegen(op, "TERF");
} else if (op->op.same_as(tl::ascend_tanh())) {
  UnaryVecOpCodegen(op, "TTANH");
```

同时把 `operation_config.h` 的 workspace 配置改成：

```cpp
{tl::ascend_erf().get(), {2, true, true}},
{tl::ascend_tanh().get(), {2, true, true}},
```

并在 `GetPTOWorkspaceSpec()` 明确是否需要 workspace。

原因：

```text
PTO 的 UnaryVecOpCodegen 只打印 op(dst, src)，和 AscendC 高级数学接口不是同一套 tmp 语义。
没有确认宏和 workspace 之前不要标支持。
```

## 3. 测试方案

### 3.1 编译安装

```bash
cd /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend
USE_ASCEND=true ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0 pip install -e .
```

### 3.2 新增 elementwise runtime 测试

文件：

```text
testing/python/language/test_tilelang_ascend_language_elementwise_erf_tanh.py
```

建议新增最小用例：

```python
import pytest
import torch

import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _torch_dtype(dtype):
    return torch.float32 if dtype == "float" else torch.float16


def _make_vec_unary(op_name, M, N, block_M, block_N, dtype):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            if op_name == "erf":
                T.tile.erf(b_ub, a_ub)
            else:
                T.tile.tanh(b_ub, a_ub)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


@pytest.mark.parametrize("op_name", ["erf", "tanh"])
@pytest.mark.parametrize("dtype", ["float", "float16"])
def test_erf_tanh_ascendc(op_name, dtype):
    M, N = 256, 256
    func = _make_vec_unary(op_name, M, N, 64, 128, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target="ascendc")

    a = torch.randn(M, N, dtype=_torch_dtype(dtype)).npu()
    torch.npu.synchronize()
    b = func(a)

    ref = torch.erf(a) if op_name == "erf" else torch.tanh(a)
    atol = 2e-3 if dtype == "float16" else 1e-5
    rtol = 2e-3 if dtype == "float16" else 1e-5
    torch.testing.assert_close(b, ref, rtol=rtol, atol=atol)
```

执行：

```bash
pytest -q testing/python/language/test_tilelang_ascend_language_elementwise_erf_tanh.py
```

### 3.3 新增 codegen 断言

同一个测试文件里可以加一个轻量检查，确认没有退回多项式拼接：

```python
def test_erf_tanh_codegen_contains_native_ascendc():
    for op_name, needle in [("erf", "AscendC::Erf"), ("tanh", "AscendC::Tanh")]:
        func = _make_vec_unary(op_name, 128, 128, 64, 128, "float")
        mod = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=pass_configs,
            target="ascendc",
        )
        src = mod.get_kernel_source()
        assert needle in src
```

如果当前 `compile()` 返回对象没有 `get_kernel_source()`，改用项目里已有的 source dump helper 或 cache 目录读取生成 `.cpp`。

### 3.4 explicit tmp 测试

文件：

```text
testing/python/language/test_tilelang_ascend_language_explicit_tmp.py
```

修改位置：显式 tmp 参数列表里，放在 `sin/cos` 附近。

新增：

```python
(ascend_tile.erf(dst, src, **tmp_arg), 2),
(ascend_tile.tanh(dst, src, **tmp_arg), 2),
```

原因：

```text
确认 tmp= 用户显式传入时插入位置是 index=2：
  dst, src, tmp, count
```

### 3.5 lower/codegen 失败边界测试

建议加一个 PTO 负面测试，避免误标支持：

```python
def test_erf_tanh_pto_unsupported():
    func = _make_vec_unary("erf", 128, 128, 64, 128, "float")
    with pytest.raises(Exception):
        tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target="pto")
```

如果后续确认并接入 PTO，则删除这个负面测试，改成 runtime 正向测试。

## 4. 文档补充

文件：

```text
docs/TileLang-Ascend Programming Guide.md
```

在 `T.tile.exp` / 一元 vector op 说明附近补充：

```markdown
- `T.tile.erf(dst, src, tmp=None)`:

  按元素计算误差函数，等价于 `dst = erf(src)`。AscendC 后端使用
  `AscendC::Erf`，支持 `float16` / `float`。该接口需要临时 UB
  workspace；通常无需手动传 `tmp`，也可以通过 `tmp=` 传入完整 UB
  scratch buffer。

- `T.tile.tanh(dst, src, tmp=None)`:

  按元素计算双曲正切，等价于 `dst = tanh(src)`。AscendC 后端使用
  `AscendC::Tanh`，支持 `float16` / `float`。该接口需要临时 UB
  workspace；通常无需手动传 `tmp`，也可以通过 `tmp=` 传入完整 UB
  scratch buffer。
```

## 5. 验证清单

修改完成后按这个顺序验证：

```bash
cd /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend

USE_ASCEND=true ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0 pip install -e .

pytest -q testing/python/language/test_tilelang_ascend_language_explicit_tmp.py
pytest -q testing/python/language/test_tilelang_ascend_language_elementwise_erf_tanh.py
```

额外建议跑已有 elementwise 和 tail 相关测试：

```bash
pytest -q testing/python/language/test_tilelang_ascend_language_elementwise.py
pytest -q testing/python/language/test_tilelang_ascend_language_tail_block.py
pytest -q testing/python/language/test_tilelang_ascend_language_tail_mask_codegen.py
```

期望：

```text
T.tile.erf -> 生成 AscendC::Erf(dst, src, tmp, count)
T.tile.tanh -> 生成 AscendC::Tanh(dst, src, tmp, count)
float / float16 runtime 结果分别对齐 torch.erf / torch.tanh
target="pto" 第一版明确 unsupported
```

## 6. 常见风险

1. 不要只改 `ascend_tile.py` 和 `codegen_ascend.cc`。

```text
Erf/Tanh 需要 tmp，必须补 workspace policy。
否则容易生成看似能编译、但 CANN 高级数学接口临时空间不可控的代码。
```

2. 不要把 `erf/tanh` 直接加入 `UnaryTag`。

```text
普通 unary tail helper 假设 call layout 是 dst, src, count，且 mask/repeat overload 可用。
Erf/Tanh 第一版是 dst, src, tmp, count，不满足这个假设。
```

3. 不要默认支持整数或 bf16。

```text
当前文档支持范围是 half / float。
TileLang 侧先用 ICHECK 限制 float16 / float，避免模板实例化到不支持 dtype。
```

4. 不要默认支持 PTO。

```text
没有确认 PTO intrinsic 前，target="pto" 应保持 unsupported。
```

## 7. 第三阶段：补齐 bfloat16 支持

这一阶段按当前仓库已经改过的 `erf/tanh` 链路继续补，不再单独设计 fallback。目标是让下面两个 case 直接跑通：

```text
T.tile.erf(dst_bf16_ub, src_bf16_ub)
T.tile.tanh(dst_bf16_ub, src_bf16_ub)
```

最终生成代码应直接包含：

```text
AscendC::Erf(...)
AscendC::Tanh(...)
```

并且 bfloat16 case 的生成源码里不应出现：

```text
CAST_LOW2HIGH
CAST_HIGH2LOW
```

### 7.1 当前已改文件状态

你当前已经改过这些文件：

```text
tilelang/language/ascend_tile.py
src/op/ascend.h
src/op/ascend.cc
src/transform/common/operation_config.h
src/transform/allocate_tmp_buffer.cc
src/target/codegen_ascend.cc
src/transform/ascend_vid_reduction.cc
src/transform/ascend_tail_mask_propagation.cc
testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py
```

其中主链路已经是：

```text
T.tile.erf / tanh
  -> tl.ascend_erf / tl.ascend_tanh
  -> InjectTmpBuffer 在第 2 个位置插入 tmp
  -> AscendC::Erf(dst, src, tmp, count)
  -> AscendC::Tanh(dst, src, tmp, count)
```

所以 bfloat16 要补的是：

```text
1. workspace 估算允许 bfloat16。
2. codegen 不排除 bfloat16。
3. tail / vid pass 识别 erf/tanh。
4. 测试脚本加入 bfloat16，并检查生成源码没有 cast。
```

### 7.2 修改 Python API

文件：

```text
tilelang/language/ascend_tile.py
```

当前文件末尾已经加了 `advanced_unary_op`、`erf`、`tanh`。保持这个实现即可，但建议把最后的 `######` 和无换行整理掉。

最终保留为：

```python
def advanced_unary_op(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    op: str,
    *,
    tmp: Buffer | BufferRegion | None = None,
):
    if isinstance(dst, BufferRegion):
        dst_ptr, dst_extent = _handle_buffer_region(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")
        dst_extent = dst.shape

    if isinstance(src0, BufferRegion):
        src0_ptr, src0_extent = _handle_buffer_region(src0, "r")
    else:
        src0_ptr = src0.access_ptr("r")
        src0_extent = src0.shape

    size_0 = math.prod(dst_extent)
    size_1 = math.prod(src0_extent)
    assert size_0 == size_1, "size must be same"

    return _call_intrin_with_optional_tmp(op, [dst_ptr, src0_ptr, size_0], 2, tmp)


def erf(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    *,
    tmp: Buffer | BufferRegion | None = None,
):
    return advanced_unary_op(dst, src0, "erf", tmp=tmp)


def tanh(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    *,
    tmp: Buffer | BufferRegion | None = None,
):
    return advanced_unary_op(dst, src0, "tanh", tmp=tmp)
```

不要在 Python 层限制 dtype。

原因：

```text
T.tile 是 ascend_tile 模块别名。
dtype 是否支持应由 C++ workspace/codegen/AscendC 编译共同约束。
这样 bfloat16 不会在 Python API 阶段被拦住。
```

### 7.3 修改 builtin 注册

文件 1：

```text
src/op/ascend.h
```

在 `ascend_cumsum()` 后面保留：

```cpp
TVM_DLL const Op &ascend_erf();
TVM_DLL const Op &ascend_tanh();
```

文件 2：

```text
src/op/ascend.cc
```

在 `TIR_DEFINE_TL_BUILTIN(ascend_cumsum)` 后面保留：

```cpp
TIR_DEFINE_TL_BUILTIN(ascend_erf)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tanh)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

这里必须是 4 个输入：

```text
dst, src, tmp, count
```

### 7.4 修改 workspace op 配置

文件：

```text
src/transform/common/operation_config.h
```

修改位置 1：`GetOperationConfig()` 的 op 表里，放在 `tl.ascend_cumsum` 附近。

加入：

```cpp
{"tl.ascend_erf", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
{"tl.ascend_tanh", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
```

修改位置 2：`GetWorkspaceOpConfigs()`，放在 `ascend_gather()` 后或其它 workspace op 附近。

加入：

```cpp
{tl::ascend_erf().get(), {2, true, false}},
{tl::ascend_tanh().get(), {2, true, false}},
```

原因：

```text
tmp_arg_index = 2，表示 InjectTmpBuffer 把 tmp 插到 dst/src 后：
  tl.ascend_erf(dst, src, tmp, count)
  tl.ascend_tanh(dst, src, tmp, count)

ascendc_supported = true。
pto_supported = false，当前不接 PTO。
```

### 7.5 修改 workspace 大小估算并放开 bfloat16

文件：

```text
src/transform/allocate_tmp_buffer.cc
```

文件顶部需要有：

```cpp
#include <tvm/arith/analyzer.h>
```

在 `EstimateAscendCBroadcastWorkspaceBytes` 后面加入或替换成下面版本：

```cpp
int64_t EstimateAscendCErfWorkspaceBytes(const CallNode *call) {
  const DataType dtype = GetAccessPtrDtype(call->args[1]);
  const int64_t src_bytes = GetAccessPtrBytes(call->args[1]);

  const bool is_half = dtype.is_float() && dtype.bits() == 16;
  const bool is_bfloat16 = dtype.is_bfloat16();
  const bool is_float = dtype.is_float() && dtype.bits() == 32;

  ICHECK(is_half || is_bfloat16 || is_float)
      << "AscendC Erf only supports float16/bfloat16/float32, got " << dtype;

  const int64_t factor = is_float ? 3 : 8;
  return factor * std::max<int64_t>(src_bytes, 256);
}

int64_t EstimateAscendCTanhWorkspaceBytes(const CallNode *call) {
  const DataType dtype = GetAccessPtrDtype(call->args[1]);
  const int64_t src_bytes = GetAccessPtrBytes(call->args[1]);

  const bool is_half = dtype.is_float() && dtype.bits() == 16;
  const bool is_bfloat16 = dtype.is_bfloat16();
  const bool is_float = dtype.is_float() && dtype.bits() == 32;

  ICHECK(is_half || is_bfloat16 || is_float)
      << "AscendC Tanh only supports float16/bfloat16/float32, got " << dtype;

  const int64_t factor = is_float ? 1 : 4;
  return factor * std::max<int64_t>(src_bytes, 256);
}
```

在 `GetAscendCWorkspaceSpec()` 里，放在 `pow` 分支后面：

```cpp
if (call->op.same_as(tl::ascend_erf())) {
  return RequireWorkspace(byte_dtype, EstimateAscendCErfWorkspaceBytes(call));
}
if (call->op.same_as(tl::ascend_tanh())) {
  return RequireWorkspace(byte_dtype, EstimateAscendCTanhWorkspaceBytes(call));
}
```

这里 bfloat16 按 half 的 workspace factor 处理：

```text
Erf:
  float32 factor = 3
  float16/bfloat16 factor = 8

Tanh:
  float32 factor = 1
  float16/bfloat16 factor = 4
```

### 7.6 修改 AscendC codegen

文件：

```text
src/target/codegen_ascend.cc
```

在 `CodeGenTileLangAscend::VisitExpr_(const CallNode *op, std::ostream &os)` 里，放在 `ascend_exp` / `ascend_sin` / `ascend_cos` 这些一元 vector op 附近。

加入：

```cpp
} else if (op->op.same_as(tl::ascend_erf())) {
  TrigOpCodegen(op, "AscendC::Erf");
} else if (op->op.same_as(tl::ascend_tanh())) {
  TrigOpCodegen(op, "AscendC::Tanh");
```

不要写成：

```cpp
UnaryVecOpCodegen(op, "AscendC::Erf");
UnaryVecOpCodegen(op, "AscendC::Tanh");
```

原因：

```text
UnaryVecOpCodegen 适配的是 dst, src, count。
Erf/Tanh 当前是 dst, src, tmp, count，必须走和 sin/cos 一样的 TrigOpCodegen 打印方式。
```

bfloat16 这里不需要额外改 dtype 映射，当前 `getType()` 已经支持：

```cpp
return "bfloat16_t";
```

所以生成代码会直接推导 `LocalTensor<bfloat16_t>`。

### 7.7 修改 VID reduction allow-list

文件：

```text
src/transform/ascend_vid_reduction.cc
```

在 `IsTileOp()` 的 `tile_ops` 里，找到一元 op 列表：

```cpp
"tl.ascend_exp", "tl.ascend_ln", "tl.ascend_abs",
"tl.ascend_reciprocal", "tl.ascend_sqrt", "tl.ascend_rsqrt",
```

改成：

```cpp
"tl.ascend_exp", "tl.ascend_ln", "tl.ascend_abs",
"tl.ascend_erf", "tl.ascend_tanh",
"tl.ascend_reciprocal", "tl.ascend_sqrt", "tl.ascend_rsqrt",
```

原因：

```text
AscendVidReduction 会在 vid 拆分 UB 行时改 tile op 的最后一个 count 参数。
erf/tanh 的最后一个参数也是 count，不加这里会导致 vid 拆分后仍按原 tile 大小计算。
```

### 7.8 修改 tail mask propagation

文件：

```text
src/transform/ascend_tail_mask_propagation.cc
```

在 `TryRewriteVectorOp()` 里，放在 `ascend_reduce()` 分支后、`ascend_cast()` 分支前。

加入：

```cpp
if (call->op.same_as(ascend_erf()) || call->op.same_as(ascend_tanh())) {
  if (call->args.size() >= 2) {
    PropagateUnaryShape(call->args[0], call->args[1]);
  }
  return Stmt();
}
```

不要把 `erf/tanh` 加进 `UnaryTag()`。

原因：

```text
UnaryTag/RewriteUnary 假设 call layout 是：
  dst, src, count

erf/tanh 的实际 layout 是：
  dst, src, tmp, count

另外 Erf/Tanh 高级数学接口文档没有 mask/repeat overload。
所以第一版只传播 tail mask，不重写成 tail_unary。
```

### 7.9 `common.h` 不需要为 erf/tanh 增加 helper

文件：

```text
src/tl_templates/ascend/common.h
```

这次不要新增 `tl::ascend::erf` / `tl::ascend::tanh` helper。

原因：

```text
codegen_ascend.cc 直接打印 AscendC::Erf / AscendC::Tanh。
不会走 common.h 里的 tl::ascend helper。
```

只需要确认不要在本次新增代码里写：

```cpp
!std::is_same_v<T, bfloat16_t>
```

当前 common.h 里已有的 `!std::is_same_v<T, bfloat16_t>` 出现在 transpose fast path 附近，和 erf/tanh 无关，不要动。

### 7.10 修改直接执行测试脚本，加入 bfloat16

文件：

```text
testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py
```

修改位置 1：dtype 列表。

把：

```python
ALL_DTYPES = ["float", "float16"]
```

改成：

```python
ALL_DTYPES = ["float", "float16", "bfloat16"]
```

修改位置 2：`torch_dtype()`。

替换成：

```python
def torch_dtype(dtype: str):
    import torch  # pylint: disable=import-outside-toplevel

    if dtype == "float":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype: {dtype}")
```

修改位置 3：`tolerance()`。

替换成：

```python
def tolerance(dtype: str) -> tuple[float, float]:
    if dtype == "float16":
        return 3e-3, 3e-3
    if dtype == "bfloat16":
        return 2e-2, 2e-2
    return 1e-5, 1e-5
```

修改位置 4：`print_kernel_source()` 增加 cast 检查。

替换成：

```python
def print_kernel_source(kernel, op_name: str) -> None:
    subline("Generated kernel source")
    src = kernel.get_kernel_source()
    print(src)

    expected = "AscendC::Erf" if op_name == "erf" else "AscendC::Tanh"
    print(f"\nsource contains {expected}: {expected in src}")
    print(f"source contains CAST_LOW2HIGH : {'CAST_LOW2HIGH' in src}")
    print(f"source contains CAST_HIGH2LOW : {'CAST_HIGH2LOW' in src}")
```

### 7.11 验证命令

重新编译：

```bash
cd /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend
cmake --build build -j2
```

先看 lower/codegen：

```bash
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf bfloat16 ascendc --lower-only
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh bfloat16 ascendc --lower-only
```

再跑 runtime：

```bash
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf bfloat16 ascendc --no-source
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh bfloat16 ascendc --no-source
```

打开 source 确认没有 cast：

```bash
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf bfloat16 ascendc
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh bfloat16 ascendc
```

期望：

```text
source contains AscendC::Erf: True
source contains AscendC::Tanh: True
source contains CAST_LOW2HIGH : False
source contains CAST_HIGH2LOW : False
RESULT: PASS op=erf, dtype=bfloat16
RESULT: PASS op=tanh, dtype=bfloat16
```

### 7.12 如果 bfloat16 编译失败，看这两个点

如果报 workspace dtype unsupported：

```text
说明 allocate_tmp_buffer.cc 里 EstimateAscendCErfWorkspaceBytes /
EstimateAscendCTanhWorkspaceBytes 还没有放开 dtype.is_bfloat16()。
```

如果报 AscendC 模板不支持 `bfloat16_t`：

```text
说明底层 CANN 当前版本的 AscendC::Erf / AscendC::Tanh 不能直接吃 bfloat16_t。
这种情况下框架侧不能实现 native bf16，只能回到手动或 pass 自动 cast。
但按本方案目标，先按 native bf16 路径接，失败再看 CANN 报错决定是否需要 fallback。
```

## 8. 第四阶段：支持 Erf/Tanh 算法配置参数

这一阶段解决 GELU 相消点精度问题。

当前实现没有传算法配置参数。实际链路是：

```text
T.tile.erf(dst, src)
  -> tl.ascend_erf(dst, src, count)
  -> InjectTmpBuffer
  -> tl.ascend_erf(dst, src, tmp, count)
  -> AscendC::Erf(dst, src, tmp, count)

T.tile.tanh(dst, src)
  -> tl.ascend_tanh(dst, src, count)
  -> InjectTmpBuffer
  -> tl.ascend_tanh(dst, src, tmp, count)
  -> AscendC::Tanh(dst, src, tmp, count)
```

这等价于使用 AscendC 默认模板参数：

```text
Erf:
  isReuseSource = false
  config = defaultErfConfig
  algo = AscendC::ErfAlgo::PADE_APPROXIMATION

Tanh:
  isReuseSource = false
  config = DEFAULT_TANH_CONFIG
  algo = AscendC::TanhAlgo::INTRINSIC
```

问题：

```text
Erf 默认 PADE_APPROXIMATION 在 z≈-3.92 这类 GELU 相消点会把结果截断到 -1.0。
这样 1 + erf(z) 直接变成 0，GELU 输出丢失 torch.erf 保留的尾部残差。
```

目标：

```text
T.tile.erf(dst, src, algo="subsection_polynomial")
  -> AscendC::Erf<T, false, erf_config>(dst, src, tmp, count)

T.tile.tanh(dst, src, algo="subsection_compensation")
  -> AscendC::Tanh<T, false, tanh_config>(dst, src, tmp, count)
```

### 8.1 修改 Python API，增加 algo 参数

文件：

```text
tilelang/language/ascend_tile.py
```

替换当前 `advanced_unary_op` 为：

```python
def advanced_unary_op(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    op: str,
    *,
    tmp: Buffer | BufferRegion | None = None,
    algo: str | None = None,
):
    if isinstance(dst, BufferRegion):
        dst_ptr, dst_extent = _handle_buffer_region(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")
        dst_extent = dst.shape

    if isinstance(src0, BufferRegion):
        src0_ptr, src0_extent = _handle_buffer_region(src0, "r")
    else:
        src0_ptr = src0.access_ptr("r")
        src0_extent = src0.shape

    size_0 = math.prod(dst_extent)
    size_1 = math.prod(src0_extent)
    assert size_0 == size_1, "size must be same"

    if op == "erf":
        algo = "pade" if algo is None else algo
        allowed = {"pade", "subsection_polynomial"}
    elif op == "tanh":
        algo = "intrinsic" if algo is None else algo
        allowed = {"intrinsic", "subsection_compensation"}
    else:
        raise ValueError(f"unsupported advanced unary op: {op}")

    if algo not in allowed:
        raise ValueError(f"T.tile.{op} algo must be one of {sorted(allowed)}, got {algo}")

    return _call_intrin_with_optional_tmp(
        op,
        [dst_ptr, src0_ptr, algo, size_0],
        2,
        tmp,
    )
```

替换当前 `erf` 为：

```python
def erf(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    *,
    tmp: Buffer | BufferRegion | None = None,
    algo: str = "pade",
):
    """Performs element-wise error function: dst = erf(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
        tmp: Optional complete UB scratch storage.
        algo: "pade" for AscendC default high-performance PADE approximation,
            or "subsection_polynomial" for higher precision.
    """
    return advanced_unary_op(dst, src0, "erf", tmp=tmp, algo=algo)
```

替换当前 `tanh` 为：

```python
def tanh(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    *,
    tmp: Buffer | BufferRegion | None = None,
    algo: str = "intrinsic",
):
    """Performs element-wise hyperbolic tangent: dst = tanh(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
        tmp: Optional complete UB scratch storage.
        algo: "intrinsic" for AscendC default high-performance implementation,
            or "subsection_compensation" for higher precision.
    """
    return advanced_unary_op(dst, src0, "tanh", tmp=tmp, algo=algo)
```

生成的 TIR 参数顺序要变成：

```text
未注入 tmp:
  tl.ascend_erf(dst, src, algo, count)
  tl.ascend_tanh(dst, src, algo, count)

注入 tmp 后:
  tl.ascend_erf(dst, src, tmp, algo, count)
  tl.ascend_tanh(dst, src, tmp, algo, count)
```

注意：

```text
algo 必须放在 count 前面，让 count 仍然是最后一个参数。
AscendVidReduction 依赖“最后一个参数是 count”来做 vid 拆分。
```

### 8.2 修改 builtin 输入个数

文件：

```text
src/op/ascend.cc
```

把当前：

```cpp
TIR_DEFINE_TL_BUILTIN(ascend_erf)
    .set_num_inputs(4)
```

改成：

```cpp
TIR_DEFINE_TL_BUILTIN(ascend_erf)
    .set_num_inputs(5)
```

把当前：

```cpp
TIR_DEFINE_TL_BUILTIN(ascend_tanh)
    .set_num_inputs(4)
```

改成：

```cpp
TIR_DEFINE_TL_BUILTIN(ascend_tanh)
    .set_num_inputs(5)
```

完整代码：

```cpp
TIR_DEFINE_TL_BUILTIN(ascend_erf)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tanh)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

原因：

```text
最终 call layout 从 dst, src, tmp, count 变成 dst, src, tmp, algo, count。
```

### 8.3 workspace 配置不需要改

文件：

```text
src/transform/common/operation_config.h
```

保持：

```cpp
{"tl.ascend_erf", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
{"tl.ascend_tanh", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
```

保持：

```cpp
{tl::ascend_erf().get(), {2, true, false}},
{tl::ascend_tanh().get(), {2, true, false}},
```

原因：

```text
tmp 位置仍然是 index=2。
algo 是 StringImm 编译期参数，不是 buffer，不需要 read/write 配置。
count 仍然是最后一个参数。
```

### 8.4 修改 codegen 入口

文件 1：

```text
src/target/codegen_ascend.h
```

在 `TrigOpCodegen` 声明后面新增：

```cpp
void ErfOpCodegen(const CallNode *op);

void TanhOpCodegen(const CallNode *op);
```

文件 2：

```text
src/target/codegen_ascend.cc
```

在 `VisitExpr_` 里，把当前：

```cpp
} else if (op->op.same_as(tl::ascend_erf())) {
  TrigOpCodegen(op, "AscendC::Erf");
} else if (op->op.same_as(tl::ascend_tanh())) {
  TrigOpCodegen(op, "AscendC::Tanh");
}
```

替换成：

```cpp
} else if (op->op.same_as(tl::ascend_erf())) {
  ErfOpCodegen(op);
} else if (op->op.same_as(tl::ascend_tanh())) {
  TanhOpCodegen(op);
}
```

原因：

```text
TrigOpCodegen 只会按参数顺序打印普通函数调用，无法打印 ErfConfig/TanhConfig 模板参数。
```

### 8.5 新增 ErfOpCodegen / TanhOpCodegen

文件：

```text
src/target/codegen_ascend.cc
```

放在 `TrigOpCodegen` 后面。

新增：

```cpp
void CodeGenTileLangAscend::ErfOpCodegen(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 5U)
      << "tl.ascend_erf expects dst, src, tmp, algo, count";

  const DataType dtype = GetAccessPtrDtype(op->args[1].as<CallNode>());
  const std::string type = getType(dtype);
  const std::string dst = PrintBufferOffset(op->args[0].as<CallNode>());
  const std::string src = PrintBufferOffset(op->args[1].as<CallNode>());
  const std::string tmp = PrintBufferOffset(op->args[2].as<CallNode>());
  const std::string algo = Downcast<StringImm>(op->args[3])->value;
  const std::string count = PrintExpr(op->args[4]);

  if (algo == "pade") {
    this->PrintIndent();
    this->stream << "AscendC::Erf(" << dst << ", " << src << ", " << tmp
                 << ", " << count << ");\n";
    return;
  }

  ICHECK_EQ(algo, "subsection_polynomial")
      << "Unsupported T.tile.erf algo: " << algo;

  this->PrintIndent();
  this->stream << "{\n";
  {
    int scope = this->BeginScope();
    this->PrintIndent();
    this->stream
        << "static constexpr AscendC::ErfAlgo erf_algo = "
        << "AscendC::ErfAlgo::SUBSECTION_POLYNOMIAL_APPROXIMATION;\n";
    this->PrintIndent();
    this->stream
        << "static constexpr AscendC::ErfConfig erf_config = {erf_algo};\n";
    this->PrintIndent();
    this->stream << "AscendC::Erf<" << type << ", false, erf_config>(" << dst
                 << ", " << src << ", " << tmp << ", " << count << ");\n";
    this->EndScope(scope);
  }
  this->PrintIndent();
  this->stream << "}\n";
}
```

新增：

```cpp
void CodeGenTileLangAscend::TanhOpCodegen(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 5U)
      << "tl.ascend_tanh expects dst, src, tmp, algo, count";

  const DataType dtype = GetAccessPtrDtype(op->args[1].as<CallNode>());
  const std::string type = getType(dtype);
  const std::string dst = PrintBufferOffset(op->args[0].as<CallNode>());
  const std::string src = PrintBufferOffset(op->args[1].as<CallNode>());
  const std::string tmp = PrintBufferOffset(op->args[2].as<CallNode>());
  const std::string algo = Downcast<StringImm>(op->args[3])->value;
  const std::string count = PrintExpr(op->args[4]);

  if (algo == "intrinsic") {
    this->PrintIndent();
    this->stream << "AscendC::Tanh(" << dst << ", " << src << ", " << tmp
                 << ", " << count << ");\n";
    return;
  }

  ICHECK_EQ(algo, "subsection_compensation")
      << "Unsupported T.tile.tanh algo: " << algo;

  this->PrintIndent();
  this->stream << "{\n";
  {
    int scope = this->BeginScope();
    this->PrintIndent();
    this->stream
        << "static constexpr AscendC::TanhAlgo tanh_algo = "
        << "AscendC::TanhAlgo::SUBSECTION_COMPENSATION;\n";
    this->PrintIndent();
    this->stream
        << "static constexpr AscendC::TanhConfig tanh_config = {tanh_algo};\n";
    this->PrintIndent();
    this->stream << "AscendC::Tanh<" << type << ", false, tanh_config>(" << dst
                 << ", " << src << ", " << tmp << ", " << count << ");\n";
    this->EndScope(scope);
  }
  this->PrintIndent();
  this->stream << "}\n";
}
```

说明：

```text
默认 algo 仍然打印 AscendC::Erf / AscendC::Tanh 普通调用。
高精度 algo 才打印带 config 的模板调用。
外面包一层 {}，避免一个 kernel 里多次调用时 static constexpr 变量重名。
```

### 8.6 修改 tail mask propagation 注释

文件：

```text
src/transform/ascend_tail_mask_propagation.cc
```

当前逻辑可以不改，但注释里的参数布局要改。

把：

```cpp
//   dst(0), src(1), tmp(2), count(3)
```

改成：

```cpp
//   dst(0), src(1), tmp(2), algo(3), count(4)
```

逻辑保持：

```cpp
if (call->op.same_as(ascend_erf()) || call->op.same_as(ascend_tanh())) {
  if (call->args.size() >= 2) {
    PropagateUnaryShape(call->args[0], call->args[1]);
  }
  return Stmt();
}
```

### 8.7 AscendVidReduction 不需要额外改

文件：

```text
src/transform/ascend_vid_reduction.cc
```

保持 `IsTileOp()` 里已经加入：

```cpp
"tl.ascend_erf", "tl.ascend_tanh",
```

因为 `algo` 放在 `count` 前面，所以最后一个参数仍然是 count，`ModifyTileOpSize()` 还能继续工作。

### 8.8 更新直接执行测试脚本，支持 algo 参数

文件：

```text
testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py
```

该脚本要能直接 `python` 执行，不依赖 pytest 输出。最终脚本需要支持：

```text
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc subsection_polynomial
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh float ascendc subsection_compensation
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc --lower-only
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc subsection_polynomial --gelu-erf-tail --no-source
```

注意：测试脚本里第四个位置参数只用于显式测试 config 算法。默认算法不传：

```text
erf  默认就是 PADE，不要传 pade。
tanh 默认就是 INTRINSIC，不要传 intrinsic。
```

文件顶部增加 dtype/algo 配置：

```python
ALL_OPS = ["erf", "tanh"]
ALL_DTYPES = ["float", "float16", "bfloat16"]
ALL_TARGETS = ["ascendc"]
DEFAULT_ALGO = {
    "erf": "pade",
    "tanh": "intrinsic",
}
VALID_ALGOS = {
    "erf": ["subsection_polynomial"],
    "tanh": ["subsection_compensation"],
}
```

`make_kernel` 签名改成：

```python
def make_kernel(
    op_name: str,
    dtype: str,
    rows: int = 8,
    cols: int = 64,
    algo: str | None = None,
):
```

`T.tile.erf/tanh` 调用处按 `algo` 是否显式传入分支：

```python
if op_name == "erf":
    if algo is None:
        T.tile.erf(out_ub, src_ub)
    else:
        T.tile.erf(out_ub, src_ub, algo=algo)
else:
    if algo is None:
        T.tile.tanh(out_ub, src_ub)
    else:
        T.tile.tanh(out_ub, src_ub, algo=algo)
```

`lower_case` / `runtime_case` 也加 `algo` 参数，并传给 `make_kernel`：

```python
func = make_kernel(op_name, dtype, algo=algo)
```

`print_kernel_source()` 里增加 source marker，确认 codegen 走到 native AscendC 和指定算法：

```python
expected = "AscendC::Erf" if op_name == "erf" else "AscendC::Tanh"
expected_algo = resolve_algo(op_name, algo)
print(f"source contains {expected}: {expected in src}")
print(f"source contains CAST_LOW2HIGH : {'CAST_LOW2HIGH' in src}")
print(f"source contains CAST_HIGH2LOW : {'CAST_HIGH2LOW' in src}")
if op_name == "erf":
    print(
        "source contains SUBSECTION_POLYNOMIAL_APPROXIMATION : "
        f"{'SUBSECTION_POLYNOMIAL_APPROXIMATION' in src}"
    )
if op_name == "tanh":
    print(
        "source contains SUBSECTION_COMPENSATION : "
        f"{'SUBSECTION_COMPENSATION' in src}"
    )
print(f"requested algo: {expected_algo}")
```

`parse_args()` 支持第四个位置参数 `algo`，同时支持 `--lower-only`、`--no-source`、`--gelu-erf-tail`：

```python
def resolve_algo(op_name: str, algo: str | None) -> str:
    return DEFAULT_ALGO[op_name] if algo is None else algo


def format_algo(op_name: str, algo: str | None) -> str:
    return f"default({DEFAULT_ALGO[op_name]})" if algo is None else algo


def validate_algo(op_name: str, algo: str | None) -> None:
    if algo is None:
        return
    if algo not in VALID_ALGOS[op_name]:
        raise SystemExit(
            f"unknown explicit algo {algo} for {op_name}; omit the algo argument "
            f"for default {DEFAULT_ALGO[op_name]}, or use one of {VALID_ALGOS[op_name]}"
        )
```

### 8.9 新增 GELU 相消点专项测试

文件：

```text
testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py
```

新增 `--gelu-erf-tail` 模式，专门验证：

```text
z ≈ -3.9174
```

最小检查逻辑：

```python
def print_gelu_tail_from_erf(src, got_erf_f32, expected_erf_f32) -> bool:
    import torch

    tail_z = src[0, 7].detach().to(torch.float32)
    tail_x = tail_z * math.sqrt(2.0)
    got_tail_erf = got_erf_f32[0, 7].detach()
    expected_tail_erf = expected_erf_f32[0, 7].detach()
    got_gelu = 0.5 * tail_x * (1.0 + got_tail_erf)
    expected_gelu = 0.5 * tail_x * (1.0 + expected_tail_erf)
    gelu_abs_err = (got_gelu - expected_gelu).abs()
    clipped = bool(torch.eq(got_tail_erf, torch.tensor(-1.0, device=got_tail_erf.device)).item())

    print("gelu tail x     :", tail_x.detach().cpu())
    print("gelu tail got   :", got_gelu.detach().cpu())
    print("gelu tail expect:", expected_gelu.detach().cpu())
    print("gelu tail abs   :", gelu_abs_err.detach().cpu())
    print(f"erf tail clipped: {clipped}")
    return clipped
```

`make_input()` 中固定放入相消点：

```python
data[0, 7] = -3.9174
```

普通 `erf` runtime case 也调用 `print_gelu_tail_from_erf()`，这样不用额外参数也能看到 GELU 相消后的值。

`--gelu-erf-tail` 的判定规则：

```text
默认 PADE，不传第四个参数：
  got 很可能是 -1.0
  这是已知失败，不作为通过标准。

algo="subsection_polynomial":
  got 应该接近 torch.erf(z)，至少不能直接截断成 -1.0。
```

新增已知失败提示函数，直接指出当前失败属于哪个后端限制：

```python
def print_known_compile_hint(op_name: str, algo: str | None, err: Exception) -> None:
    msg = str(err)
    resolved_algo = resolve_algo(op_name, algo)
    if op_name == "erf" and resolved_algo == "subsection_polynomial":
        print(
            "known compile hint: current JIT target uses --npu-arch=dav-2201; "
            "CANN 9.0.0 exposes ErfConfig only for 3510/5102/3003/3113 branches."
        )
    if op_name == "tanh" and resolved_algo == "subsection_compensation":
        print(
            "known compile hint: current JIT target uses --npu-arch=dav-2201; "
            "CANN 9.0.0 exposes TanhConfig only for 3510/5102/3003/3113 branches."
        )
    if (
        "got bfloat16" in msg
        or "LocalTensor<__bf16>" in msg
        or "LocalTensor<bfloat16_t>" in msg
        or "Div supports half/float" in msg
    ):
        print(
            "known compile hint: native AscendC Erf/Tanh bfloat16 path is not "
            "accepted by CANN 9.0.0 for this target; use fp32 compute fallback."
        )
```

### 8.10 验证命令

如果只改测试脚本，不需要重新编译 C++。如果同时改了 C++ 文件，必须按下面命令重编并重新 editable install：

```bash
cd /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
bash install_ascend.sh --enable-incremental

USE_ASCEND=true ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0 pip install -e .
```

验证脚本语法：

```bash
python -m py_compile testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py
```

验证默认算法 lower-only 保持可用：

```bash
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc --lower-only --no-source
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh float ascendc --lower-only --no-source
```

验证高精度算法参数能进入 TIR/codegen：

```bash
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc subsection_polynomial --lower-only
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh float ascendc subsection_compensation --lower-only
```

当前默认 JIT 命令是 `--npu-arch=dav-2201`。在 CANN 9.0.0 中，`ErfConfig/TanhConfig` 只在 3510/5102/3003/3113 分支暴露；所以上述两个高精度 lower-only 在 dav-2201 下预期会失败，并打印：

```text
known compile hint: current JIT target uses --npu-arch=dav-2201; CANN 9.0.0 exposes ErfConfig only for 3510/5102/3003/3113 branches.
known compile hint: current JIT target uses --npu-arch=dav-2201; CANN 9.0.0 exposes TanhConfig only for 3510/5102/3003/3113 branches.
```

验证 bfloat16 native 路径：

```bash
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf bfloat16 ascendc --lower-only --no-source
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh bfloat16 ascendc --lower-only --no-source
```

当前 C++ workspace 检查会在 `allocate_tmp_buffer.cc` 拦住 bf16，输出类似：

```text
AscendC Erf only supports float16/bfloat16/float32, got bfloat16
AscendC Tanh only supports float16/bfloat16/float32, got bfloat16
known compile hint: native AscendC Erf/Tanh bfloat16 path is not accepted by CANN 9.0.0 for this target; use fp32 compute fallback.
```

验证 GELU 相消点：

```bash
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc subsection_polynomial --gelu-erf-tail --no-source
```

重点看：

```text
tail got erf
tail torch erf
gelu tail got
gelu tail expect
erf tail clipped
```

### 8.11 注意事项

1. `algo` 必须是编译期字符串参数。

```text
不要让 algo 变成 runtime tensor/scalar。
AscendC::ErfConfig / TanhConfig 是模板参数，必须在 codegen 阶段静态确定。
```

2. `algo` 必须放在 `count` 前面。

```text
AscendVidReduction 当前默认修改最后一个参数作为 tile size。
如果把 algo 放最后，会把 StringImm 当 count 改，直接出错。
```

3. bfloat16 和高精度算法是两件事。

```text
CANN 9.0.0 下 AscendC::Erf/Tanh<bfloat16_t> 已经验证编译失败。
高精度 subsection_polynomial 主要解决 float/float16 的尾部相消精度。
bf16 仍需要 fp32 compute fallback。
```
