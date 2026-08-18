# reduce_abssum / reduce_absmax / cumsum Ascend 接入实施方案

## 1. 第一阶段：reduce_abssum / reduce_absmax 接 AscendC

这一阶段目标是让：

```text
T.reduce_abssum(src_ub, out_ub, dim=-1)
T.reduce_absmax(src_ub, out_ub, dim=-1)
```

不要再走通用 `tl.reduce`，而是和 `reduce_sum/max/min` 一样走 Ascend 专用链路：

```text
tilelang/language/reduce_ascend.py
  -> tl.ascend_reduce
  -> InjectTmpBuffer 插入 tmp_ub
  -> codegen_ascend.cc::ReduceOpCodegen
  -> tl::ascend::reduce_abssum / reduce_absmax
  -> src/tl_templates/ascend/common.h
```

### 1.1 修改 Python 前端

文件：

```text
tilelang/language/reduce_ascend.py
```

修改位置 1：文件顶部 `__all__`。

把 `reduce_abssum` 和 `reduce_absmax` 加进去：

```python
__all__ = [
    "reduce",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_abssum",
    "reduce_absmax",
]
```

原因：

```text
tilelang/language/__init__.py 里先 import reduce.py，再 from reduce_ascend import *。
只有放进 reduce_ascend.py 的 __all__，T.reduce_abssum / T.reduce_absmax 才会覆盖 reduce.py 的通用实现。
```

修改位置 2：放在 `reduce_sum` 函数后面。

新增 `reduce_abssum`：

```python
def reduce_abssum(
    buffer: Buffer | BufferRegion,
    out: Buffer | BufferRegion,
    dim: int = -1,
    *args,
    real_shape=_REDUCE_KWARG_SENTINEL,
):
    parsed_clear, parsed_real_shape = _parse_reduce_optional_args(
        "reduce_abssum",
        args,
        clear=True,
        real_shape=real_shape,
    )
    if parsed_clear is not True:
        raise ValueError("reduce_abssum requires clear=True in the first implementation")
    legalized_dim = _legalize_reduce_dim(_get_buffer_extent(buffer), dim)
    return _reduce_with_clear(
        buffer,
        out,
        "reduce_abssum",
        legalized_dim,
        True,
        parsed_real_shape,
    )
```

新增 `reduce_absmax`：

```python
def reduce_absmax(
    buffer: Buffer | BufferRegion,
    out: Buffer | BufferRegion,
    dim: int = -1,
    *args,
    clear=_REDUCE_KWARG_SENTINEL,
    real_shape=_REDUCE_KWARG_SENTINEL,
):
    parsed_clear, parsed_real_shape = _parse_reduce_optional_args(
        "reduce_absmax",
        args,
        clear=clear,
        real_shape=real_shape,
    )
    if parsed_clear is not True:
        raise ValueError("reduce_absmax requires clear=True in the first implementation")
    legalized_dim = _legalize_reduce_dim(_get_buffer_extent(buffer), dim)
    return _reduce_with_clear(
        buffer,
        out,
        "reduce_absmax",
        legalized_dim,
        parsed_clear,
        parsed_real_shape,
    )
```

原因：

```text
_reduce_with_clear 会生成 tl.ascend_reduce。
这样 parser 之后的 TIR 会从 T.reduce(..., "abssum", ...) 变成 T.ascend_reduce("reduce_abssum<...>", ...)。
通用 ReduceOp::Lower 只支持 local.fragment，不支持 shared.ub，所以 UB 路径必须绕开 tl.reduce。
```

### 1.2 修改 AscendC helper

文件：

```text
src/tl_templates/ascend/common.h
```

修改位置：放在现有 `reduce_sum` / `reduce_max` / `reduce_min` helper 附近。建议放在 `reduce_sum` 后面、`reduce_max` 前面。

新增一个连续 abs helper：

```cpp
template <typename T>
CATLASS_DEVICE void abs_contiguous(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor,
                                   uint32_t count) {
  constexpr uint32_t kOneRepeatBytes = 256;
  constexpr uint32_t kOneRepeatElems = kOneRepeatBytes / sizeof(T);
  uint32_t repeatTime = count / kOneRepeatElems;
  uint32_t tail = count % kOneRepeatElems;

  if (repeatTime > 0) {
    uint64_t mask = kOneRepeatElems;
    AscendC::Abs(dstTensor, srcTensor, mask, repeatTime, {1, 1, 8, 8});
  }

  if (tail > 0) {
    uint64_t mask = tail;
    AscendC::Abs(
        dstTensor[repeatTime * kOneRepeatElems],
        srcTensor[repeatTime * kOneRepeatElems],
        mask,
        1,
        {1, 1, 8, 8});
  }
}
```

新增 `reduce_abssum`：

```cpp
template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void reduce_abssum(LocalTensor<T> const &dstTensor,
                                  LocalTensor<T> const &srcTensor,
                                  LocalTensor<uint8_t> const &sharedTmpBuffer,
                                  bool clear = true) {
  (void)clear;

  constexpr uint32_t srcCount = M * N;
  constexpr uint32_t absTmpBytes = srcCount * sizeof(T);

  LocalTensor<T> absTmp =
      const_cast<LocalTensor<uint8_t>&>(sharedTmpBuffer).template ReinterpretCast<T>();
  LocalTensor<uint8_t> reduceTmp =
      const_cast<LocalTensor<uint8_t>&>(sharedTmpBuffer)[absTmpBytes];

  abs_contiguous<T>(absTmp, srcTensor, srcCount);

  uint32_t shape[] = {M, N};
  if constexpr (dim == -1) {
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, absTmp, reduceTmp, shape, true);
  } else {
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, absTmp, reduceTmp, shape, true);
  }
}
```

新增 `reduce_absmax`：

```cpp
template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void reduce_absmax(LocalTensor<T> const &dstTensor,
                                  LocalTensor<T> const &srcTensor,
                                  LocalTensor<uint8_t> const &sharedTmpBuffer,
                                  bool clear = true) {
  (void)clear;

  constexpr uint32_t srcCount = M * N;
  constexpr uint32_t absTmpBytes = srcCount * sizeof(T);

  LocalTensor<T> absTmp =
      const_cast<LocalTensor<uint8_t>&>(sharedTmpBuffer).template ReinterpretCast<T>();
  LocalTensor<uint8_t> reduceTmp =
      const_cast<LocalTensor<uint8_t>&>(sharedTmpBuffer)[absTmpBytes];

  abs_contiguous<T>(absTmp, srcTensor, srcCount);

  uint32_t shape[] = {M, N};
  if constexpr (dim == -1) {
    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, absTmp, reduceTmp, shape, true);
  } else {
    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, absTmp, reduceTmp, shape, true);
  }
}
```

原因：

```text
AscendC 没有直接用当前 TileLang reduce wrapper 表达 abs reduce。
第一版组合成 Abs + ReduceSum/ReduceMax，语义最直观。
common.h 会被生成 kernel 直接 include，不能使用 TVM 的 ICHECK/LOG 宏。
```

### 1.3 修改 AscendC tmp buffer 大小

文件：

```text
src/transform/allocate_tmp_buffer.cc
```

修改位置：`createTmpBuffer_` 里处理 `tl::ascend_reduce()` 的分支。

逻辑要从普通 reduce 的：

```text
tmp_size = src_count * dtype_bytes
```

改成：

```text
reduce_abssum / reduce_absmax:
  tmp_size = abs_tmp_size + reduce_tmp_size
           = src_count * dtype_bytes + src_count * dtype_bytes
```

要点：

```cpp
std::string reduce_name = Downcast<StringImm>(call->args[0])->value;
bool need_abs_tmp =
    reduce_name.find("reduce_abssum") != std::string::npos ||
    reduce_name.find("reduce_absmax") != std::string::npos;
```

原因：

```text
helper 里 sharedTmpBuffer 前半段给 absTmp，后半段给 ReduceSum/ReduceMax 的 reduceTmp。
如果这里只按普通 reduce 申请 tmp，runtime 编译可能过，但执行会踩 tmp 空间。
```

### 1.4 AscendC codegen

文件：

```text
src/target/codegen_ascend.cc
```

这一步通常不用改。

原因：

```text
CodeGenTileLangAscend::ReduceOpCodegen 已经是通用打印：
  op_name = "tl::ascend::" + StringImm

所以只要 TIR 是：
  T.ascend_reduce("reduce_abssum<float, 8, 64, -1>", ...)

就会自动生成：
  tl::ascend::reduce_abssum<float, 8, 64, -1>(...)
```

### 1.5 验证

重新构建：

```bash
cd /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend
USE_ASCEND=true ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0 pip install -e .
```

检查 lower/codegen：

```bash
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py reduce_abssum ascendc
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py reduce_absmax ascendc
```

检查 runtime：

```bash
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py --runtime reduce_abssum ascendc
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py --runtime reduce_absmax ascendc
```

期望：

```text
reduce_abssum -> torch.abs(src).sum(dim=-1)
reduce_absmax -> torch.abs(src).amax(dim=-1)
```

## 2. 第二阶段：reduce_abssum / reduce_absmax 接 PTO

这一阶段目标是让 PTO 后端也能消费同一个 `tl.ascend_reduce("reduce_abssum/absmax<...>")`。

PTO 没有原生 `ABSREDUCE` 时，第一版组合：

```text
reduce_abssum:
  TABS(abs_tmp, src)
  TROWSUM / TCOLSUM(dst, abs_tmp, reduce_tmp)

reduce_absmax:
  TABS(abs_tmp, src)
  TROWMAX / TCOLMAX(dst, abs_tmp)
```

### 2.1 修改 PTO ReduceKind

文件：

```text
src/target/codegen_ascend_pto.h
```

修改位置：`CodeGenTileLangAscendPto` 类里的 `ReduceKind`。

把：

```cpp
enum class ReduceKind { SUM, MAX, MIN };
```

改成：

```cpp
enum class ReduceKind { SUM, MAX, MIN, ABSSUM, ABSMAX };
```

原因：

```text
ParseReduceOpInfo 需要把 reduce_abssum / reduce_absmax 解析成独立 kind。
后续 codegen 再把 ABSSUM 映射成 SUM，把 ABSMAX 映射成 MAX。
```

### 2.2 修改 PTO reduce 解析

文件：

```text
src/target/codegen_ascend_pto.cc
```

修改位置：`CodeGenTileLangAscendPto::ParseReduceOpInfo`。

把 reduce type 判断写成这个顺序：

```cpp
if (op_name.find("reduce_abssum") != std::string::npos) {
  info.kind = ReduceKind::ABSSUM;
} else if (op_name.find("reduce_absmax") != std::string::npos) {
  info.kind = ReduceKind::ABSMAX;
} else if (op_name.find("reduce_sum") != std::string::npos) {
  info.kind = ReduceKind::SUM;
} else if (op_name.find("reduce_max") != std::string::npos) {
  info.kind = ReduceKind::MAX;
} else if (op_name.find("reduce_min") != std::string::npos) {
  info.kind = ReduceKind::MIN;
} else {
  ICHECK(false) << "Unsupported reduce type: " << op_name;
}
```

原因：

```text
PTO 当前只认识 SUM/MAX/MIN，不加这一步会在 ParseReduceOpInfo 阶段直接报 unsupported reduce type。
abs kind 放前面，避免字符串匹配顺序造成误判。
```

### 2.3 修改 PTO codegen 组合逻辑

文件：

```text
src/target/codegen_ascend_pto.cc
```

修改位置 1：anonymous namespace，放在 `ParseConstBoolArg` 后面。

新增：

```cpp
using ReduceKind = CodeGenTileLangAscendPto::ReduceKind;

bool IsAbsReduceKind(ReduceKind kind) {
  return kind == ReduceKind::ABSSUM || kind == ReduceKind::ABSMAX;
}

ReduceKind BaseReduceKind(ReduceKind kind) {
  if (kind == ReduceKind::ABSSUM) {
    return ReduceKind::SUM;
  }
  if (kind == ReduceKind::ABSMAX) {
    return ReduceKind::MAX;
  }
  return kind;
}
```

原因：

```text
ABSSUM/ABSMAX 只是 codegen 组合 kind。
真正调用 PTO reduce 宏时，还是 TROWSUM/TCOLSUM 或 TROWMAX/TCOLMAX。
```

修改位置 2：`GetReduceMergeOpName`。

在函数开头加：

```cpp
kind = BaseReduceKind(kind);
```

原因：

```text
clear=False 时会先 reduce 到临时 out，再 merge 回原 dst。
ABSSUM 要用 TADD merge，ABSMAX 要用 TMAX merge。
```

修改位置 3：`GetReduceOpName`。

在函数开头加：

```cpp
kind = BaseReduceKind(kind);
```

原因：

```text
避免 unordered_map 里找 ABSSUM/ABSMAX，因为 map 只需要保存 SUM/MAX/MIN 到 PTO 宏的映射。
```

修改位置 4：`CodegenColReduce`。

把所有：

```cpp
if (op_info.kind == ReduceKind::SUM) {
```

改成：

```cpp
ReduceKind base_kind = BaseReduceKind(op_info.kind);
if (base_kind == ReduceKind::SUM) {
```

并且 `GetReduceOpName` 使用 `base_kind`：

```cpp
std::string op_name = GetReduceOpName(base_kind, ReduceDirection::COL);
```

原因：

```text
TCOLSUM 需要 tmp 参数。
reduce_abssum dim=0 最终会调用 TCOLSUM，如果还判断 op_info.kind == SUM，会少传 tmp。
```

修改位置 5：`ReduceOpCodegen`。

放在这行之后：

```cpp
ShapeInfo tmp = GetSliceInfo(op->args[3].as<CallNode>());
```

新增组合逻辑：

```cpp
auto emit_abs_if_needed = [&](const ShapeInfo &input_src,
                              ShapeInfo *reduce_tmp) -> ShapeInfo {
  if (!IsAbsReduceKind(op_info.kind)) {
    return input_src;
  }

  const int32_t elem_bytes = GetTypeLen(input_src.type);
  const int32_t abs_tmp_bytes =
      input_src.slice_row * input_src.slice_col * elem_bytes;

  ShapeInfo abs_tmp = input_src;
  abs_tmp.first_addr = tmp.first_addr;
  abs_tmp.offset = "0";
  abs_tmp.type = input_src.type;
  abs_tmp.ub_name = GetTempVarName(input_src.ub_name + "_abs_tmp");
  abs_tmp.is_slice = false;
  CreateUbVariableND(abs_tmp.ub_name, abs_tmp);

  std::string src_name = ResolveUbSliceName(input_src);
  this->PrintIndent();
  this->stream << "TABS(" << abs_tmp.ub_name << ", " << src_name << ");\n";

  *reduce_tmp = tmp;
  reduce_tmp->first_addr =
      tmp.first_addr + tir::make_const(tmp.first_addr.dtype(), abs_tmp_bytes);
  reduce_tmp->offset = "0";

  return abs_tmp;
};

ShapeInfo reduce_tmp = tmp;
ShapeInfo reduce_src = emit_abs_if_needed(src, &reduce_tmp);

ReduceOpInfo base_info = op_info;
base_info.kind = BaseReduceKind(op_info.kind);
```

然后把后续 reduce 调用里的：

```cpp
CodegenRowReduce(op_info, dst, src, tmp);
CodegenColReduce(op_info, dst, src, tmp);
```

改成：

```cpp
CodegenRowReduce(base_info, dst, reduce_src, reduce_tmp);
CodegenColReduce(base_info, dst, reduce_src, reduce_tmp);
```

如果是 `clear=False` 分支，也同样把：

```cpp
CodegenRowReduce(op_info, tmp_dst, src, tmp);
CodegenColReduce(op_info, tmp_dst, src, tmp);
```

改成：

```cpp
CodegenRowReduce(base_info, tmp_dst, reduce_src, reduce_tmp);
CodegenColReduce(base_info, tmp_dst, reduce_src, reduce_tmp);
```

原因：

```text
ReduceOpCodegen 同时拿得到 src、dst、tmp、clear，是最适合组合 TABS + reduce 的地方。
tmp 的前半段作为 abs_tmp，后半段作为原 PTO reduce 所需 reduce_tmp。
CodegenRowReduce/CodegenColReduce 保持负责“打一条 PTO reduce 宏”，不要在里面再拆 abs。
```

### 2.4 修改 PTO tmp buffer 大小

文件：

```text
src/transform/allocate_tmp_buffer.cc
```

修改位置：`GetPTOTmpBufferSize_` 里处理 `tl::ascend_reduce()` 的分支。

在把 `reduce_sum/max/min` 映射成 `TROWSUM/TCOLSUM/...` 的地方，先加 abs reduce 判断：

```cpp
bool need_abs_tmp =
    op_name.find("reduce_abssum") != std::string::npos ||
    op_name.find("reduce_absmax") != std::string::npos;

if (op_name.find("reduce_abssum") != std::string::npos) {
  op_name = (mode == "row") ? "TROWSUM" : "TCOLSUM";
} else if (op_name.find("reduce_absmax") != std::string::npos) {
  op_name = (mode == "row") ? "TROWMAX" : "TCOLMAX";
} else if (op_name.find("reduce_sum") != std::string::npos) {
  op_name = (mode == "row") ? "TROWSUM" : "TCOLSUM";
} else if (op_name.find("reduce_max") != std::string::npos) {
  op_name = (mode == "row") ? "TROWMAX" : "TCOLMAX";
} else if (op_name.find("reduce_min") != std::string::npos) {
  op_name = (mode == "row") ? "TROWMIN" : "TCOLMIN";
} else {
  ICHECK(false) << "not support reduce type: " << op_name;
}
```

计算 tmp size 时：

```text
abs_tmp_size = valid_row * AlignReduceOutputCols(valid_col, dtype_bytes) * dtype_bytes
total_tmp_size = abs_tmp_size + reduce_tmp_size
```

原因：

```text
PTO TileUbDataND 的 col 会按 32B 对齐，所以 abs_tmp 不能只按 valid_col 算。
codegen 里会把 tmp.first_addr + abs_tmp_bytes 作为 reduce_tmp 起点，所以这里必须申请总大小。
```

### 2.5 PTO reduce 验证

重新构建：

```bash
cd /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend
USE_ASCEND=true ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0 pip install -e .
```

检查 codegen：

```bash
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py reduce_abssum pto
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py reduce_absmax pto
```

期望 source：

```text
reduce_abssum dim=-1:
  TABS(...)
  TROWSUM(...)

reduce_absmax dim=-1:
  TABS(...)
  TROWMAX(...)
```

如果后续测试 `dim=0`：

```text
reduce_abssum dim=0:
  TABS(...)
  TCOLSUM(...)

reduce_absmax dim=0:
  TABS(...)
  TCOLMAX(...)
```

## 3. 第三阶段：cumsum 接 AscendC

这一阶段只处理 AscendC 后端，PTO 先不管。目标是让：

```text
T.cumsum(src_ub, out_ub, dim=-1)
```

从通用 `tl.cumsum` 改成 Ascend 专用链路：

```text
tilelang/language/reduce_ascend.py
  -> tl.ascend_cumsum
  -> InjectTmpBuffer 插入 tmp_ub 和 last_row_ub
  -> codegen_ascend.cc::CumSumOpCodegen
  -> tl::ascend::cumsum<...>(dst, last_row, src, tmp)
  -> src/tl_templates/ascend/common.h
  -> AscendC::CumSum
```

第一版只支持：

```text
target = ascendc
2D UB tensor
dim = -1 / 1 或 dim = -2 / 0
reverse = False
dtype = float / half
```

### 3.1 注册 C++ op：`tl.ascend_cumsum`

#### 3.1.1 修改 `src/op/ascend.h`

文件：

```text
src/op/ascend.h
```

当前位置：`ascend_reduce()` 声明在 `ascend_gather()` 下面，当前大约是：

```cpp
TVM_DLL const Op &ascend_gather();

TVM_DLL const Op &ascend_reduce();

TVM_DLL const Op &ascend_block_reduce_max();
```

修改方式：在 `TVM_DLL const Op &ascend_reduce();` 下边加：

```cpp
TVM_DLL const Op &ascend_cumsum();
```

改完变成：

```cpp
TVM_DLL const Op &ascend_gather();

TVM_DLL const Op &ascend_reduce();

TVM_DLL const Op &ascend_cumsum();

TVM_DLL const Op &ascend_block_reduce_max();
```

原因：

```text
Python 前端会调用 tir.op.Op.get("tl.ascend_cumsum")。
这个名字必须在 C++ 侧声明和注册，否则构造 TIR Call 时找不到 op。
```

#### 3.1.2 修改 `src/op/ascend.cc`

文件：

```text
src/op/ascend.cc
```

当前位置：`TIR_DEFINE_TL_BUILTIN(ascend_reduce)` 在 `TIR_DEFINE_TL_BUILTIN(ascend_gather)` 下边，大约是：

```cpp
TIR_DEFINE_TL_BUILTIN(ascend_reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_block_reduce_max)
```

修改方式：在 `TIR_DEFINE_TL_BUILTIN(ascend_reduce)` 这一段下边加：

```cpp
TIR_DEFINE_TL_BUILTIN(ascend_cumsum)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

改完变成：

```cpp
TIR_DEFINE_TL_BUILTIN(ascend_reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_cumsum)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_block_reduce_max)
```

原因：

```text
set_num_inputs(4) 对应 tmp 注入前的 Python/TIR 参数：
  op_name, dst_ptr, src_ptr, reverse

InjectTmpBuffer 后会变成：
  op_name, dst_ptr, src_ptr, tmp_ptr, last_row_ptr, reverse

和 ascend_reduce 一样，后续 pass 会插入额外参数。
```

### 3.2 修改 Python 前端：让 `T.cumsum` 走 Ascend 专用 op

文件：

```text
tilelang/language/reduce_ascend.py
```

#### 3.2.1 修改 `__all__`

当前位置：文件顶部 `__all__` 里已经有 reduce 相关 API，例如：

```python
__all__ = [
    "reduce",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_abssum",
    "reduce_absmax",
]
```

修改方式：在 `"reduce_absmax",` 下边加：

```python
"cumsum",
```

原因：

```text
tilelang/language/__init__.py 会先导入 reduce.py 的通用 cumsum，
再 from reduce_ascend import *。
只有 reduce_ascend.py 的 __all__ 里导出 cumsum，T.cumsum 才会覆盖通用实现。
```

#### 3.2.2 新增 Ascend 专用 `cumsum`

当前位置：放在 `reduce_absmax(...)` 函数定义下边。

新增：

```python
def cumsum(
    src: Buffer | BufferRegion,
    dst: Buffer | BufferRegion | None = None,
    dim: int = 0,
    reverse: bool = False,
):
    if dst is None:
        dst = src

    shape = _get_buffer_extent(src)
    if len(shape) != 2:
        raise ValueError("Ascend cumsum first implementation only supports 2D UB tensor")
    if dim < 0:
        dim = len(shape) + dim
    if dim not in (0, 1):
        raise ValueError("Ascend cumsum only supports dim=0/-2 or dim=1/-1")
    if reverse:
        raise ValueError("Ascend cumsum reverse=True is not implemented in the first version")

    dtype = _dtype(src)
    op_name = f"cumsum<{dtype}, {shape[0]}, {shape[1]}, {dim}>"
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_cumsum"),
        op_name,
        dst.access_ptr("w"),
        src.access_ptr("r"),
        reverse,
    )
```

原因：

```text
通用 reduce.py::cumsum 生成 tl.cumsum。
tl.cumsum 是通用 op，当前不适合 AscendC UB codegen。
这里改成生成 tl.ascend_cumsum，后续由 AscendC 专用 codegen 消费。
```

### 3.3 修改 operation config：让 pass 识别 `ascend_cumsum`

文件：

```text
src/transform/common/operation_config.h
```

#### 3.3.1 修改 pipeline/resource 配置

当前位置：`kOpPipelineStages` 里有 `tl.ascend_reduce`，大约是：

```cpp
{"tl.ascend_reduce",
 {{{1, "write"}, {2, "read"}, {3, "read"}}, "PIPE_V"}},
{"tl.ascend_block_reduce_max", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
```

修改方式：在 `tl.ascend_reduce` 这一项下边加：

```cpp
{"tl.ascend_cumsum",
 {{{1, "write"}, {2, "read"}, {3, "read"}, {4, "write"}}, "PIPE_V"}},
```

原因：

```text
tmp 注入后参数顺序是：
  0 op_name
  1 dst_ptr      write
  2 src_ptr      read
  3 tmp_ptr      scratch，按 read 处理即可
  4 last_row_ptr write
  5 reverse

pipeline/sync pass 需要知道这个 op 的读写参数，否则后续自动同步和资源分析可能漏掉 cumsum。
```

#### 3.3.2 修改 `ascendc_tmp_arg_ops`

当前位置：`ascendc_tmp_arg_ops` 里有 `ascend_reduce`，大约是：

```cpp
const std::unordered_map<const tvm::OpNode *, int64_t> ascendc_tmp_arg_ops = {
    {tl::ascend_clamp().get(), 3},
    {tl::ascend_clamp_max().get(), 3},
    {tl::ascend_clamp_min().get(), 3},
    {tl::ascend_reduce().get(), 3},
    {tl::ascend_sort().get(), 3},
```

修改方式：在 `{tl::ascend_reduce().get(), 3},` 下边加：

```cpp
{tl::ascend_cumsum().get(), 3},
```

原因：

```text
CallNodeCollector 只收集 ascendc_tmp_arg_ops 里的 op。
不加这一项，InjectTmpBuffer 不会处理 ascend_cumsum，也就不会创建/插入 tmp 参数。
插入位置 3 表示把 tmp 插到 src_ptr 后面：
  原始: op_name, dst, src, reverse
  插入 tmp 后: op_name, dst, src, tmp, reverse

cumsum 还需要 last_row，所以后面会在 CallNodeModifier 里专门处理 ascend_cumsum，插入 tmp 和 last_row 两个参数。
```

### 3.4 修改 `src/transform/allocate_tmp_buffer.cc`

这是 AscendC cumsum 最关键的部分。要做三件事：

```text
1. 创建 cumsum 用的 tmp_ub。
2. 额外创建 cumsum_last_row_ub。
3. 把 tmp_ub 和 cumsum_last_row_ub 都插到 tl.ascend_cumsum 参数里。
```

#### 3.4.1 扩展 `CallNodeModifier::Modify`

文件：

```text
src/transform/allocate_tmp_buffer.cc
```

当前位置：`class CallNodeModifier` 开头，当前 `Modify` 大约是：

```cpp
static Stmt Modify(PrimFunc f, Target target, Buffer &tmp_buffer,
                   Array<Buffer> &tmp_buffers,
                   Buffer &reduce_out_tmp_buffer) {
  CallNodeModifier modifier;
  ...
  modifier.tmp_buf_ = tmp_buffer;
  modifier.tmp_bufs_ = tmp_buffers;
  modifier.reduce_out_tmp_buf_ = reduce_out_tmp_buffer;
  return modifier.AddTmpArg(f->body);
}
```

修改方式：把函数签名扩展为：

```cpp
static Stmt Modify(PrimFunc f, Target target, Buffer &tmp_buffer,
                   Array<Buffer> &tmp_buffers,
                   Buffer &reduce_out_tmp_buffer,
                   Buffer &cumsum_last_row_buffer) {
```

然后在：

```cpp
modifier.reduce_out_tmp_buf_ = reduce_out_tmp_buffer;
```

下边加：

```cpp
modifier.cumsum_last_row_buf_ = cumsum_last_row_buffer;
```

原因：

```text
CallNodeModifier 负责重写 Call 参数。
它原来只能拿到 tmp_buf_ / tmp_bufs_ / reduce_out_tmp_buf_。
cumsum 需要再插入 last_row buffer，所以必须把 cumsum_last_row_buffer 传进来。
```

#### 3.4.2 修改 `CallNodeModifier` 里的 `VisitExpr_`

当前位置：在 `class CallNodeModifier` 内部搜索这个函数签名。注意它是类内部定义，所以源码里没有 `CallNodeModifier::` 前缀：

```cpp
PrimExpr VisitExpr_(const CallNode *op) override {
```

在函数里的这段下面：

```cpp
if (const auto *op_node = op->op.as<OpNode>()) {
  if (tmp_arg_ops_.count(op_node) > 0) {
    int64_t tmp_buffer_param_offset = tmp_arg_ops_.at(op_node);
```

紧接着加 cumsum 特判，放在 `NeedReduceOutputTmp(op)` 之前：

```cpp
if (op->op.same_as(tl::ascend_cumsum())) {
  ICHECK(cumsum_last_row_buf_.defined())
      << "ascend_cumsum expects cumsum_last_row buffer.";
  Array<PrimExpr> new_args = op->args;
  new_args = InsertExprAt_(
      new_args, tmp_buffer_param_offset, MakeAccessPtrFromBuffer_(tmp_buf_, 1));
  new_args = InsertExprAt_(
      new_args, tmp_buffer_param_offset + 1,
      MakeAccessPtrFromBuffer_(cumsum_last_row_buf_, 2));
  return Call(op->dtype, op->op, new_args, op->span);
}
```

最终逻辑顺序应该是：

```cpp
int64_t tmp_buffer_param_offset = tmp_arg_ops_.at(op_node);
if (op->op.same_as(tl::ascend_cumsum())) {
  ...
}
if (NeedReduceOutputTmp(op)) {
  ...
}
```

原因：

```text
普通 CallNodeAddTmp 只能插一个 tmp。
ascend_cumsum 需要插两个参数：
  tmp_ptr
  last_row_ptr

原始参数:
  op_name, dst_ptr, src_ptr, reverse

插入后:
  op_name, dst_ptr, src_ptr, tmp_ptr, last_row_ptr, reverse
```

#### 3.4.3 给 `CallNodeModifier` 增加成员变量

当前位置：`CallNodeModifier` 类尾部成员变量处，当前大约是：

```cpp
Buffer tmp_buf_;
Array<Buffer> tmp_bufs_;
Buffer reduce_out_tmp_buf_;
std::string target_;
std::unordered_map<const tvm::OpNode *, int64_t> tmp_arg_ops_;
```

修改方式：在 `Buffer reduce_out_tmp_buf_;` 下边加：

```cpp
Buffer cumsum_last_row_buf_;
```

原因：

```text
VisitExpr_ 里要调用 MakeAccessPtrFromBuffer_(cumsum_last_row_buf_, 2)。
```

#### 3.4.4 修改 `TmpBufferInjector::TmpBufferInject`

当前位置：`TmpBufferInjector::TmpBufferInject` 里调用 `CallNodeModifier::Modify` 的地方，当前大约是：

```cpp
new_body = CallNodeModifier::Modify(f, target, injector.tmp_buf_,
                                    injector.tmp_bufs_,
                                    injector.reduce_out_tmp_buf_);
```

修改方式：在最后追加 `injector.cumsum_last_row_buf_`：

```cpp
new_body = CallNodeModifier::Modify(f, target, injector.tmp_buf_,
                                    injector.tmp_bufs_,
                                    injector.reduce_out_tmp_buf_,
                                    injector.cumsum_last_row_buf_);
```

原因：

```text
TmpBufferInjector 负责创建 buffer。
CallNodeModifier 负责把 buffer access_ptr 插到 call 参数里。
这里要把创建出来的 cumsum_last_row_buf_ 传给 CallNodeModifier。
```

#### 3.4.5 在 `tilelang_root` block 里追加 last_row buffer

当前位置：`TmpBufferInjector::VisitStmt_(const BlockRealizeNode *node)`，当前创建 tmp 的代码是：

```cpp
tmp_buf_ = createTmpBuffer_(op->alloc_buffers);
if (tmp_buf_.defined()) {
  new_alloc_buffers.push_back(tmp_buf_);
}

if ("pto" == target_) {
```

修改方式：在 `tmp_buf_` push 完之后、`if ("pto" == target_)` 之前加：

```cpp
cumsum_last_row_buf_ = createCumSumLastRowBuffer_(op->alloc_buffers);
if (cumsum_last_row_buf_.defined()) {
  new_alloc_buffers.push_back(cumsum_last_row_buf_);
}
```

改完顺序：

```cpp
tmp_buf_ = createTmpBuffer_(op->alloc_buffers);
if (tmp_buf_.defined()) {
  new_alloc_buffers.push_back(tmp_buf_);
}

cumsum_last_row_buf_ = createCumSumLastRowBuffer_(op->alloc_buffers);
if (cumsum_last_row_buf_.defined()) {
  new_alloc_buffers.push_back(cumsum_last_row_buf_);
}

if ("pto" == target_) {
```

原因：

```text
last_row 是一个真正的 alloc_buffer，必须加入 tilelang_root block 的 alloc_buffers。
否则后面 MakeAccessPtrFromBuffer_ 虽然能构造 access_ptr，但这个 buffer 不在 IR 分配列表里。
```

#### 3.4.6 新增 `createCumSumLastRowBuffer_`

当前位置：放在 `createTmpBuffer_(Array<Buffer> alloc_buffers)` 函数后面。

也就是这段后面：

```cpp
Buffer createTmpBuffer_(Array<Buffer> alloc_buffers) {
  ...
}
```

新增：

```cpp
Buffer createCumSumLastRowBuffer_(Array<Buffer> alloc_buffers) {
  for (size_t i = 0; i < calls_.size(); i++) {
    const CallNode *call = calls_[i].get();
    if (!call->op.same_as(tl::ascend_cumsum())) {
      continue;
    }

    std::string op_name = Downcast<StringImm>(call->args[0])->value;
    auto template_params = ExtractTemplateParamsForSliceBuffer(op_name);
    int64_t m = std::get<0>(template_params);
    int64_t n = std::get<1>(template_params);
    int64_t last_row_size = std::max(m, n);

    const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
    std::string src_buffer_name =
        src_access_ptr->args[1].as<VarNode>()->name_hint;
    const BufferNode *src_buffer_node =
        GetBufferNodeByName_(alloc_buffers, src_buffer_name);
    ICHECK(src_buffer_node);

    Var tmp_buf("cumsum_last_row",
                PointerType(PrimType(src_buffer_node->dtype), "shared.ub"));
    return Buffer(tmp_buf, src_buffer_node->dtype,
                  {IntImm(DataType::Int(32), last_row_size)}, {}, PrimExpr(),
                  "cumsum_last_row", -1, 0, BufferType::kDefault);
  }
  return Buffer();
}
```

注意：`ExtractTemplateParamsForSliceBuffer` 对 `cumsum<float, M, N, dim>` 返回的是：

```text
std::get<0> -> M
std::get<1> -> N
std::get<2> -> dim
```

原因：

```text
AscendC::CumSum 需要 lastRowTensor。
第一版用 max(M, N) 保守申请，避免 dim=0/1 时 lastRowTensor 形状不够。
后续如果确认 AscendC 文档里的精确大小，可以再改成 dim 相关大小。
```

#### 3.4.7 修改 `createTmpBuffer_` 的 size 计算

当前位置：`GetAscendCTmpBufferSize_` 中遍历 `calls_` 的分支。现在已经有很多：

```cpp
if (call->op.same_as(tl::ascend_clamp())) {
  ...
} else if (call->op.same_as(tl::ascend_reduce())) {
  ...
} else if (call->op.same_as(tl::ascend_sort())) {
  ...
}
```

修改方式：在 `tl::ascend_reduce()` 分支后面、`tl::ascend_sort()` 分支前面加 `tl::ascend_cumsum()` 分支：

```cpp
else if (call->op.same_as(tl::ascend_cumsum())) {
  const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
  std::string src_buffer_name =
      src_access_ptr->args[1].as<VarNode>()->name_hint;
  const BufferNode *src_buffer_node =
      GetBufferNodeByName_(alloc_buffers, src_buffer_name);
  ICHECK(src_buffer_node);

  int64_t src_count = Downcast<IntImm>(src_access_ptr->args[3])->value;
  int64_t tmp_shape_size = src_count * src_buffer_node->dtype.bytes() * 4;

  if (tmp_shape_size > shape_size) {
    shape = {IntImm(DataType::Int(32), tmp_shape_size)};
    shape_size = tmp_shape_size;
  }
}
```

原因：

```text
createTmpBuffer_ 只创建一个全局 tmp_ub，并按所有需要 tmp 的 call 取最大 size。
cumsum 需要 sharedTmpBuffer，所以必须参与 GetAscendCTmpBufferSize_。

第一版先用 src_count * dtype_bytes * 4 保守申请。
CANN CumSum 文档说明内部涉及精度转换，可能需要额外临时空间；后续可替换成官方 tmp size 公式。
```

#### 3.4.8 给 `TmpBufferInjector` 增加成员变量

当前位置：`TmpBufferInjector` 类尾部成员变量处，找到：

```cpp
Buffer tmp_buf_;
Array<Buffer> tmp_bufs_;
Buffer reduce_out_tmp_buf_;
std::string target_;
```

修改方式：在 `Buffer reduce_out_tmp_buf_;` 下边加：

```cpp
Buffer cumsum_last_row_buf_;
```

原因：

```text
TmpBufferInjector 创建 last_row buffer 后，需要保存到成员变量，再传给 CallNodeModifier。
```

### 3.5 修改 AscendC codegen

#### 3.5.1 修改 `src/target/codegen_ascend.h`

文件：

```text
src/target/codegen_ascend.h
```

当前位置：`ReduceOpCodegen` 声明附近：

```cpp
void GatherCodegen(const CallNode *op, const std::string &op_name);

void ReduceOpCodegen(const CallNode *op);

void BlockReduceOpCodegen(const CallNode *op, const std::string &op_name);
```

修改方式：在 `void ReduceOpCodegen(const CallNode *op);` 下边加：

```cpp
void CumSumOpCodegen(const CallNode *op);
```

原因：

```text
codegen_ascend.cc 里要新增成员函数实现，头文件必须先声明。
```

#### 3.5.2 修改 `src/target/codegen_ascend.cc` 的分发

文件：

```text
src/target/codegen_ascend.cc
```

当前位置：`VisitExpr_(const CallNode *op)` 分发里，找到：

```cpp
} else if (op->op.same_as(tl::ascend_reduce())) {
  ReduceOpCodegen(op);
} else if (op->op.same_as(tl::ascend_block_reduce_max())) {
```

修改方式：在 `ascend_reduce` 分支下边加：

```cpp
} else if (op->op.same_as(tl::ascend_cumsum())) {
  CumSumOpCodegen(op);
```

改完：

```cpp
} else if (op->op.same_as(tl::ascend_reduce())) {
  ReduceOpCodegen(op);
} else if (op->op.same_as(tl::ascend_cumsum())) {
  CumSumOpCodegen(op);
} else if (op->op.same_as(tl::ascend_block_reduce_max())) {
```

原因：

```text
CodeGenTileLangAscend::VisitExpr_ 是所有 TIR Call 到 AscendC 源码输出的分发入口。
不加这个分支，tl.ascend_cumsum 即使在 IR 里存在，也不会生成 CumSum 代码。
```

#### 3.5.3 新增 `CumSumOpCodegen`

文件：

```text
src/target/codegen_ascend.cc
```

当前位置：放在 `ReduceOpCodegen` 函数上边或下边。建议放在：

```cpp
void CodeGenTileLangAscend::ReduceOpCodegen(const CallNode *op) {
```

这一段上边。

新增：

```cpp
void CodeGenTileLangAscend::CumSumOpCodegen(const CallNode *op) {
  std::string op_name =
      "tl::ascend::" + Downcast<StringImm>(op->args[0])->value;

  auto dst = PrintBufferOffset(op->args[1].as<CallNode>());
  auto src = PrintBufferOffset(op->args[2].as<CallNode>());
  auto tmp = PrintBufferOffset(op->args[3].as<CallNode>());
  auto last_row = PrintBufferOffset(op->args[4].as<CallNode>());
  bool reverse = !is_zero(op->args[5]);
  ICHECK(!reverse) << "AscendC cumsum reverse=True is not implemented yet";

  this->PrintIndent();
  this->stream << op_name << "("
               << dst << ", "
               << last_row << ", "
               << src << ", "
               << tmp << ");\n";
}
```

原因：

```text
TIR 参数顺序：
  op_name, dst, src, tmp, last_row, reverse

AscendC helper 参数顺序：
  dst, last_row, src, tmp

所以这里要重新排列参数。
```

### 3.6 修改 AscendC helper

文件：

```text
src/tl_templates/ascend/common.h
```

当前位置：放在 reduce helper 附近。建议放在 `reduce_absmax` helper 下边，或者所有 reduce helper 后边。

新增：

```cpp
template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void cumsum(LocalTensor<T> const &dstTensor,
                           LocalTensor<T> const &lastRowTensor,
                           LocalTensor<T> const &srcTensor,
                           LocalTensor<uint8_t> const &sharedTmpBuffer) {
  constexpr bool is_last_axis = (dim == 1 || dim == -1);
  constexpr AscendC::CumSumConfig cumSumConfig{
      is_last_axis,
      false,
      true,
      AscendC::CumSumAlgorithm::CUMSUM_ALGORITHM_LINEBYLINE};
  const AscendC::CumSumInfo cumSumInfo{M, N};

  AscendC::CumSum<T, cumSumConfig>(
      dstTensor,
      lastRowTensor,
      srcTensor,
      sharedTmpBuffer,
      cumSumInfo);
}
```

原因：

```text
AscendC 官方 CumSum 的参数顺序是：
  dstTensor, lastRowTensor, srcTensor, sharedTmpBuffer, cumSumInfo

TileLang 的 dim=1/-1 对应 isLastAxis=true。
TileLang 的 dim=0/-2 对应 isLastAxis=false。
第一版 reverse=True 在 Python 和 codegen 两层都拒绝。

common.h 会被生成 kernel 直接 include，不要在这里用 TVM 的 ICHECK/LOG。
```

### 3.7 验证 AscendC cumsum

重新构建：

```bash
cd /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend
USE_ASCEND=true ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0 pip install -e .
```

检查 Python 绑定：

```bash
PYTHONPATH=/mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend:$PYTHONPATH \
python - <<'PY'
from tilelang import language as T
import inspect
print(T.cumsum)
print(T.cumsum.__module__)
print(inspect.getsourcefile(T.cumsum))
PY
```

期望：

```text
tilelang.language.reduce_ascend
```

检查 lower/codegen：

```bash
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py cumsum ascendc
```

期望 IR 里出现：

```text
T.ascend_cumsum(...)
```

期望 generated source 里出现：

```text
tl::ascend::cumsum<float, 8, 64, 1>(...)
```

检查 runtime：

```bash
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py --runtime cumsum ascendc
```

期望：

```text
output == torch.cumsum(src, dim=-1)
```

## 4. 第四阶段：cumsum 接 PTO

这一阶段目标是处理：

```text
T.cumsum(src_ub, out_ub, dim=-1)
target="pto"
```

建议分两步：

```text
第一步：先显式报错，避免继续静默丢计算语句。
第二步：确认 PTO 是否有 scan/prefix-sum 宏后，再做真正实现。
```

### 4.1 先让 PTO 明确不支持 cumsum

文件：

```text
src/target/codegen_ascend_pto.cc
```

修改位置：`VisitExpr_(const CallNode *op)` 分发里，放在 `ascend_reduce` 附近。

新增：

```cpp
} else if (op->op.same_as(tl::ascend_cumsum())) {
  LOG(FATAL) << "PTO cumsum is not implemented yet";
```

原因：

```text
当前 issue 里的问题是 cumsum 计算语句被 PTO codegen 丢弃，不报错但结果不对。
在没有 PTO scan 实现前，显式报错比静默生成错误 kernel 更安全。
```

### 4.2 如果 PTO 有原生 scan/cumsum 宏

文件 1：

```text
src/target/codegen_ascend_pto.h
```

修改位置：`ReduceOpCodegen` 声明附近。

新增：

```cpp
void CumSumOpCodegen(const CallNode *op);
```

文件 2：

```text
src/target/codegen_ascend_pto.cc
```

修改位置 1：`VisitExpr_(const CallNode *op)` 分发。

把 4.1 的显式报错改成：

```cpp
} else if (op->op.same_as(tl::ascend_cumsum())) {
  CumSumOpCodegen(op);
```

修改位置 2：新增 `CumSumOpCodegen`。

实现方式取决于 PTO ISA 是否有现成宏。预期参数和 AscendC 一致：

```text
op_name, dst_ptr, src_ptr, tmp_ptr, last_row_ptr, reverse
```

codegen 内部先解析：

```cpp
ShapeInfo dst = GetSliceInfo(op->args[1].as<CallNode>());
ShapeInfo src = GetSliceInfo(op->args[2].as<CallNode>());
ShapeInfo tmp = GetSliceInfo(op->args[3].as<CallNode>());
ShapeInfo last_row = GetSliceInfo(op->args[4].as<CallNode>());
bool reverse = !is_zero(op->args[5]);
```

第一版建议限制：

```text
reverse == false
src/dst dtype 一致
2D shape
dim 只支持 1/-1
```

如果 PTO 有类似 `TCUMSUM` 的宏，最终打印：

```cpp
TCUMSUM(dst_name, src_name, tmp_name);
```

具体宏名按 PTO ISA 文档替换。

原因：

```text
cumsum 是 scan，不是普通 elementwise，也不能用 reduce_sum 一步替代。
只有 PTO 有明确 scan/cumsum 能力时，才应该打开真实 codegen。
```

### 4.3 如果 PTO 没有原生 scan/cumsum 宏

不建议第一版手写循环模拟。

原因：

```text
PTO tile 是向量/矩阵宏抽象，手写逐元素 prefix-sum 很容易破坏性能，也容易在动态 valid shape、对齐列、dim=0、reverse=True 上出错。
```

此时保持 4.1 的显式报错即可，并在测试里把 `cumsum pto` 标成 expected fail。

### 4.4 PTO cumsum 验证

如果只是显式报错，验证：

```bash
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py cumsum pto
```

期望：

```text
PTO cumsum is not implemented yet
```

如果后续实现了 PTO scan，验证：

```bash
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py cumsum pto
```

期望 generated source 里出现真实 PTO cumsum/scan 宏，而不是只有 copy：

```text
copy_gm_to_ub
PTO cumsum/scan op
copy_ub_to_gm
```
