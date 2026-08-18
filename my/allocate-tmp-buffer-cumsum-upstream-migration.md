# cumsum 在新版 allocate_tmp_buffer.cc 上的迁移方案

本文说明如何把旧实现里的 cumsum workspace 逻辑迁移到新版上游：

旧参考文件：

```text
/mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend/src/transform/allocate_tmp_buffer.cc
```

目标新仓库：

```text
/mnt/workspace/gitCode/cann/tail-kernel/my/tilelang-ascend
```

新版上游已经重构了 tmp buffer 机制，所以旧文件中 `//////` 之间的代码不能整块复制。尤其不要把下面这些旧机制搬到新版：

```cpp
tmp_arg_ops_
pto_tmp_arg_ops
ascendc_tmp_arg_ops
tmp_bufs_
cumsum_last_row_buf_
createCumSumLastRowBuffer_
GetAscendCTmpBufferSize_
```

新版机制的核心是：

```cpp
WorkspaceOpConfig
GetWorkspaceOpConfigs()
WorkspaceSpec
GetWorkspaceSpec()
GetAscendCWorkspaceSpec()
CallNodeModifier::VisitExpr_()
GetTmpBufferSize_()
```

因此 cumsum 的迁移要接到这套新机制里。

## 1. 先恢复新版上游文件

如果当前文件里已经混进了旧实现，先恢复这两个文件到最新上游：

```bash
cd /mnt/workspace/gitCode/cann/tail-kernel/my/tilelang-ascend

git checkout origin/ascendc_pto -- \
  src/transform/allocate_tmp_buffer.cc \
  src/transform/common/operation_config.h
```

这样可以避免旧的 `tmp_arg_ops_` 和新版 `WorkspaceSpec` 混在一起。

## 2. 修改 operation_config.h

文件：

```text
src/transform/common/operation_config.h
```

### 2.1 添加 tl.ascend_cumsum 的基础 OperationConfig

在 `GetOperationConfig()` 里，找到：

```cpp
{"tl.ascend_reduce",
 {{{1, "write"}, {2, "read"}, {3, "read"}}, "PIPE_V"}},
```

在它后面添加：

```cpp
{"tl.ascend_cumsum", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
```

说明：

- Python 前端原始 cumsum call 是 `[op_name, dst, src, reverse]`。
- workspace 插入后才会变成 `[op_name, dst, src, tmp, last_row, reverse]`。
- 新版 `ResolveOperationConfig()` 对 workspace op 会根据实际 `tvm_access_ptr` 动态重建 buffer accesses，所以这里保留原始 dst/src 关系即可。

### 2.2 添加 WorkspaceOpConfig

在 `GetWorkspaceOpConfigs()` 里，找到末尾类似：

```cpp
{tl::ascend_select().get(), {3, true, true}},
{tl::ascend_gather_mask().get(), {4, true, true}},
{tl::ascend_gather().get(), {5, true, true}},
```

在下面添加：

```cpp
{tl::ascend_cumsum().get(), {3, true, false}},
```

注意必须是：

```cpp
{3, true, false}
```

不能写旧版：

```cpp
{tl::ascend_cumsum().get(), 3}
```

因为新版 value 类型是：

```cpp
struct WorkspaceOpConfig {
  int64_t tmp_arg_index;
  bool ascendc_supported;
  bool pto_supported;
};
```

这里含义是：

- `3`: workspace 插入位置在原始 args[3]，也就是 `reverse` 前面。
- `true`: AscendC 支持。
- `false`: PTO 暂不支持 cumsum。

## 3. 修改 allocate_tmp_buffer.cc

文件：

```text
src/transform/allocate_tmp_buffer.cc
```

## 3.0 先补 reduce_abssum / reduce_absmax 的模板解析

新版 `allocate_tmp_buffer.cc` 的 `ParseReduceTemplateInfo(...)` 只认识：

```cpp
reduce_sum
reduce_max
reduce_min
```

如果前端生成：

```text
reduce_abssum<float, 8, 64, -1>
reduce_absmax<float, 8, 64, -1>
```

`InjectTmpBuffer` 会在 workspace 估算阶段报：

```text
Unsupported reduce operation reduce_abssum<float, 8, 64, -1>
```

所以要先扩展 reduce kind。

### 修改位置 1：扩展 ReduceKind

找到：

```cpp
enum class ReduceKind { kSum, kMax, kMin };
```

改成：

```cpp
enum class ReduceKind { kSum, kMax, kMin, kAbsSum, kAbsMax };
```

### 修改位置 2：扩展 ParseReduceTemplateInfo

找到：

```cpp
ReduceKind kind;
if (op_name.find("reduce_sum") != std::string::npos) {
  kind = ReduceKind::kSum;
} else if (op_name.find("reduce_max") != std::string::npos) {
  kind = ReduceKind::kMax;
} else {
  ICHECK(op_name.find("reduce_min") != std::string::npos)
      << "Unsupported reduce operation " << op_name;
  kind = ReduceKind::kMin;
}
```

改成：

```cpp
ReduceKind kind;
if (op_name.find("reduce_abssum") != std::string::npos) {
  kind = ReduceKind::kAbsSum;
} else if (op_name.find("reduce_absmax") != std::string::npos) {
  kind = ReduceKind::kAbsMax;
} else if (op_name.find("reduce_sum") != std::string::npos) {
  kind = ReduceKind::kSum;
} else if (op_name.find("reduce_max") != std::string::npos) {
  kind = ReduceKind::kMax;
} else {
  ICHECK(op_name.find("reduce_min") != std::string::npos)
      << "Unsupported reduce operation " << op_name;
  kind = ReduceKind::kMin;
}
```

注意 `reduce_abssum` / `reduce_absmax` 要放在普通 `reduce_sum` / `reduce_max` 前面。

### 修改位置 3：AscendC workspace 估算增加 abs 临时区

找到：

```cpp
// The current wrapper still has a sharedTmpBuffer parameter even when the
// selected CANN branch reports zero bytes. Keep one aligned block
// for implicit calls; explicit arenas are deliberately not size-checked.
return std::max<int64_t>(bytes, 32);
```

在它前面添加：

```cpp
if (info.kind == ReduceKind::kAbsSum || info.kind == ReduceKind::kAbsMax) {
  bytes += info.rows * info.cols * dtype_bytes;
}
```

原因：

- `reduce_abssum` / `reduce_absmax` 的 helper 会先对输入做 `Abs`，写入临时区。
- 然后再把这个 abs 临时结果传给 `ReduceSum` / `ReduceMax`。
- 所以 workspace 要包含：

```text
abs tmp + reduce tmp
```

旧实现里对应的是：

```cpp
abs_tmp_size + base_reduce_tmp_size
```

新版写到 `EstimateAscendCReduceWorkspaceBytes(...)` 里。

---

目标：不要再创建独立的 `cumsum_last_row` buffer。新版做法是申请一个统一 `tmp_ub` arena，然后在 `CallNodeModifier` 里切成两个 view：

```text
tmp view      -> uint8 workspace, 传给 AscendC::CumSum sharedTmpBuffer
last_row view -> src dtype workspace, 传给 AscendC::CumSum lastRowTensor
```

这样 codegen 仍然可以看到旧 ABI：

```text
args[0] op_name
args[1] dst
args[2] src
args[3] tmp
args[4] last_row
args[5] reverse
```

### 3.1 添加 cumsum workspace layout helper

在匿名 namespace 里，放在 `RequireWorkspace(...)` 后面或 `GetAscendCWorkspaceSpec(...)` 前面。

添加：

```cpp
struct CumSumWorkspaceLayout {
  int64_t tmp_bytes;
  int64_t last_row_offset;
  int64_t last_row_bytes;
  DataType last_row_dtype;
};

CumSumWorkspaceLayout GetCumSumWorkspaceLayout(const CallNode *call) {
  ICHECK(call->op.same_as(tl::ascend_cumsum()));
  ICHECK_GE(call->args.size(), 4U) << "Malformed AscendC cumsum call.";

  const std::string op_name = Downcast<StringImm>(call->args[0])->value;
  const size_t left = op_name.find('<');
  const size_t right = op_name.rfind('>');
  ICHECK(left != std::string::npos && right != std::string::npos &&
         left < right)
      << "Failed to parse cumsum template " << op_name;

  std::vector<std::string> params;
  size_t begin = left + 1;
  while (begin < right) {
    const size_t comma = op_name.find(',', begin);
    const size_t end =
        comma == std::string::npos || comma > right ? right : comma;
    params.push_back(Trim(op_name.substr(begin, end - begin)));
    begin = end + 1;
  }
  ICHECK_EQ(params.size(), 4U) << "Failed to parse cumsum template "
                               << op_name;

  const int64_t m = ParseStaticInt(params[1], op_name);
  const int64_t n = ParseStaticInt(params[2], op_name);
  ICHECK_GT(m, 0);
  ICHECK_GT(n, 0);

  const DataType src_dtype = GetAccessPtrDtype(call->args[2]);
  const int64_t src_bytes = GetAccessPtrBytes(call->args[2]);

  // Keep the old first implementation's heuristic:
  // tmp_ub size = src element bytes * 4.
  const int64_t tmp_bytes = src_bytes * 4;

  // CANN CumSum lastRowTensor holds the last row/column result.
  const int64_t last_row_elems = std::max(m, n);
  const int64_t last_row_offset = AlignUp(tmp_bytes, 32);
  const int64_t last_row_bytes = last_row_elems * src_dtype.bytes();

  return {tmp_bytes, last_row_offset, last_row_bytes, src_dtype};
}
```

依赖说明：

- 新版 `allocate_tmp_buffer.cc` 没有 `ExtractTemplateParamsForSliceBuffer(...)`，不要复制旧 helper 调用。
- 直接复用新版文件里已经存在的 `Trim(...)` 和 `ParseStaticInt(...)` 解析模板参数。
- `AlignUp(...)` 新版 reduce workspace 逻辑已经使用过，直接复用。
- `GetAccessPtrDtype(...)` / `GetAccessPtrBytes(...)` 是新版 helper，已经在文件上方定义。

### 3.2 在 GetAscendCWorkspaceSpec 里加入 cumsum

找到：

```cpp
WorkspaceSpec GetAscendCWorkspaceSpec(const CallNode *call,
                                      const Array<Buffer> &alloc_buffers) {
```

在最后的：

```cpp
if (call->op.same_as(tl::ascend_merge_sort()) ||
    call->op.same_as(tl::ascend_select()) ||
    call->op.same_as(tl::ascend_gather_mask()) ||
    call->op.same_as(tl::ascend_gather())) {
  return NoWorkspace();
}
```

前面添加：

```cpp
if (call->op.same_as(tl::ascend_cumsum())) {
  const CumSumWorkspaceLayout layout = GetCumSumWorkspaceLayout(call);
  return RequireWorkspace(
      byte_dtype, layout.last_row_offset + layout.last_row_bytes);
}
```

说明：

- `WorkspaceSpec.primary_bytes` 对 cumsum 表示整个 arena 的总字节数。
- 前半段用于 `tmp`。
- 后半段按 dtype 重新解释成 `last_row`。

### 3.3 在 CallNodeModifier 里 special-case cumsum

找到新版 `CallNodeModifier::VisitExpr_(const CallNode *op)`，结构应类似：

```cpp
PrimExpr VisitExpr_(const CallNode *op) override {
  if (const auto *op_node = op->op.as<OpNode>()) {
    const auto config_it = GetWorkspaceOpConfigs().find(op_node);
    if (config_it != GetWorkspaceOpConfigs().end()) {
      const int64_t tmp_buffer_param_offset = config_it->second.tmp_arg_index;
      const bool has_workspace =
          HasWorkspaceOperand(op, tmp_buffer_param_offset);
      const WorkspaceSpec spec =
          GetWorkspaceSpec(op, alloc_buffers_, target_);
      ...
```

在：

```cpp
if (!spec.requires_workspace) {
  ...
}
```

后面、`if (has_workspace)` 前面添加：

```cpp
if (op->op.same_as(tl::ascend_cumsum())) {
  return CallNodeAddCumSumWorkspace(op, tmp_buffer_param_offset, spec,
                                    has_workspace);
}
```

也就是顺序应为：

```cpp
if (!spec.requires_workspace) {
  return has_workspace
             ? CallWithoutWorkspaceArgs(op, tmp_buffer_param_offset)
             : StmtExprMutator::VisitExpr_(op);
}
if (op->op.same_as(tl::ascend_cumsum())) {
  return CallNodeAddCumSumWorkspace(op, tmp_buffer_param_offset, spec,
                                    has_workspace);
}
if (has_workspace) {
  return HandleExistingTmp(op, tmp_buffer_param_offset, spec);
}
```

### 3.4 添加 CallNodeAddCumSumWorkspace helper

放在 `CallNodeAddTmp(...)` 附近即可。

添加：

```cpp
Call CallNodeAddCumSumWorkspace(const CallNode *op,
                                int64_t tmp_buffer_param_offset,
                                const WorkspaceSpec &spec,
                                bool has_workspace) {
  const CumSumWorkspaceLayout layout = GetCumSumWorkspaceLayout(op);
  ICHECK_GE(spec.primary_bytes,
            layout.last_row_offset + layout.last_row_bytes);

  const PrimExpr arena =
      has_workspace
          ? op->args[tmp_buffer_param_offset]
          : MakeAccessPtrFromBuffer_(tmp_buf_, spec.access_mask);

  Array<PrimExpr> new_args;
  for (int64_t i = 0; i < tmp_buffer_param_offset; ++i) {
    new_args.push_back(op->args[i]);
  }

  new_args.push_back(MakeAccessPtrView(arena, 0, layout.tmp_bytes,
                                       DataType::UInt(8), spec.access_mask));
  new_args.push_back(MakeAccessPtrView(arena, layout.last_row_offset,
                                       layout.last_row_bytes,
                                       layout.last_row_dtype, 2));

  const size_t skip = has_workspace ? 1 : 0;
  for (size_t i = tmp_buffer_param_offset + skip; i < op->args.size(); ++i) {
    new_args.push_back(op->args[i]);
  }
  return Call(op->dtype, op->op, new_args, op->span);
}
```

说明：

- 原始 cumsum args 是 `[op_name, dst, src, reverse]`。
- 插入后变成 `[op_name, dst, src, tmp, last_row, reverse]`。
- 如果用户已经显式提供了一个 workspace arena，`has_workspace=true` 时会把这个 arena 拆成两个 view。
- 如果没有显式 workspace，就用自动分配的 `tmp_buf_`。
- 不需要独立 `cumsum_last_row_buf_` 成员。

## 4. 不要迁移旧代码块

旧文件中这些内容不要迁移：

### 4.1 不要迁移 CallNodeModifier 旧签名

不要用：

```cpp
static Stmt Modify(PrimFunc f, Target target, Buffer &tmp_buffer,
                   Array<Buffer> &tmp_buffers,
                   Buffer &reduce_out_tmp_buffer,
                   Buffer &cumsum_last_row_buffer)
```

新版应保留：

```cpp
static Stmt Modify(PrimFunc f, Target target, Buffer &tmp_buffer,
                   Buffer &reduce_out_tmp_buffer,
                   const Array<Buffer> &alloc_buffers)
```

### 4.2 不要迁移 createCumSumLastRowBuffer_

旧版：

```cpp
Buffer createCumSumLastRowBuffer_(Array<Buffer> alloc_buffers)
```

新版不需要。last row 从统一 workspace arena 里切 view。

### 4.3 不要迁移 GetAscendCTmpBufferSize_

旧版：

```cpp
Array<PrimExpr> GetAscendCTmpBufferSize_(Array<Buffer> alloc_buffers)
```

新版统一用：

```cpp
Array<PrimExpr> GetTmpBufferSize_(Array<Buffer> alloc_buffers)
```

它会遍历 `calls_` 并调用：

```cpp
GetWorkspaceSpec(...)
```

所以 cumsum 的大小估计必须写在 `GetAscendCWorkspaceSpec(...)` 里。

## 5. codegen 和 Python 前端保持不变

当前 Python 前端：

```python
return tir.call_intrin(
    "handle",
    tir.op.Op.get("tl.ascend_cumsum"),
    op_name,
    dst.access_ptr("w"),
    src.access_ptr("r"),
    reverse,
)
```

保持不变。

当前 codegen：

```cpp
auto dst = PrintBufferOffset(op->args[1].as<CallNode>());
auto src = PrintBufferOffset(op->args[2].as<CallNode>());
auto tmp = PrintBufferOffset(op->args[3].as<CallNode>());
auto last_row = PrintBufferOffset(op->args[4].as<CallNode>());
bool reverse = !is_zero(op->args[5]);
```

也保持不变。

迁移后的 `allocate_tmp_buffer.cc` 会负责把原始 args：

```text
[op_name, dst, src, reverse]
```

改成 codegen 需要的：

```text
[op_name, dst, src, tmp, last_row, reverse]
```

## 6. 验证命令

### 6.1 检查不能有旧残留

```bash
grep -n "tmp_arg_ops_\\|tmp_bufs_\\|cumsum_last_row_buf_\\|createCumSumLastRowBuffer_\\|GetAscendCTmpBufferSize_\\|<<<<<<<\\|>>>>>>>" \
  src/transform/allocate_tmp_buffer.cc \
  src/transform/common/operation_config.h
```

正常应该没有输出。

### 6.2 检查 cumsum 配置

```bash
grep -n "tl.ascend_cumsum\\|ascend_cumsum().get" \
  src/transform/common/operation_config.h
```

应至少看到：

```cpp
{"tl.ascend_cumsum", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
{tl::ascend_cumsum().get(), {3, true, false}},
```

### 6.3 格式化

```bash
clang-format -i \
  src/transform/allocate_tmp_buffer.cc \
  src/transform/common/operation_config.h
```

### 6.4 编译

```bash
bash install_ascend.sh --enable-incremental
```

### 6.5 运行 cumsum 测试

```bash
PYTHONPATH=/mnt/workspace/gitCode/cann/tail-kernel/my/tilelang-ascend:$PYTHONPATH \
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py cumsum ascendc
```

如果脚本支持 runtime：

```bash
PYTHONPATH=/mnt/workspace/gitCode/cann/tail-kernel/my/tilelang-ascend:$PYTHONPATH \
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py --runtime cumsum ascendc
```

## 7. 常见错误对应关系

### 7.1 `WorkspaceOpConfig was not declared`

说明 `operation_config.h` 里 `struct WorkspaceOpConfig` 被删了或冲突没解好。

处理：

```bash
git checkout origin/ascendc_pto -- src/transform/common/operation_config.h
```

然后只补 cumsum 两处配置。

### 7.2 `could not convert ... {tl::ascend_cumsum().get(), 3}`

说明把旧版 map 写法搬进了新版 map。

改成：

```cpp
{tl::ascend_cumsum().get(), {3, true, false}},
```

### 7.3 `tmp_arg_ops_ was not declared`

说明旧版 `CallNodeModifier` 被搬进新版文件了。

处理：

```bash
git checkout origin/ascendc_pto -- src/transform/allocate_tmp_buffer.cc
```

然后按本文第 3 节迁移。

### 7.4 `tmp_bufs_ was not declared`

同上。新版 `CallNodeModifier::Modify` 不接收 `tmp_bufs_`。

### 7.5 `cumsum_last_row_buf_ was not declared`

说明旧版独立 last-row buffer 机制被部分搬进来了。新版不要这个成员，改用统一 workspace arena 切 view。
