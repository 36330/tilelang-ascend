# PR #1421 Change List

对比范围：

```bash
git diff --name-status origin/ascendc_pto..HEAD
```

含义：

- `A`: PR 新增文件
- `M`: PR 修改已有文件

注意：下面只列 PR merge 进来的提交改动，不包含当前工作区里你自己的未提交文件，例如 `my/`、`run_tilelang_npu.sh`、`examples/flash_attention/opprof_*`。

## 新增文件

```text
A 3rdparty/patches/README.md
A 3rdparty/patches/apply_tvm_patches.sh
A 3rdparty/patches/tvm_slice_step_fix.patch
A docs/pytest_marker_guide.md
A examples/tail_mask/example_tail_add.py
A examples/tile_kernels/mhc/head_compute_mix_kernel.py
A examples/tile_kernels/moe/moe_topk_gate.py
A src/transform/ascend_sync_insert_vs.cc
A src/transform/ascend_tail_mask_propagation.cc
A src/transform/common/ascend_tail_mask.h
A testing/python/language/test_ascend_sync_insert_vs.py
A testing/python/language/test_tilelang_ascend_language_copy_pad_value.py
A testing/python/language/test_tilelang_ascend_language_dynamic_fill.py
A testing/python/language/test_tilelang_ascend_language_gemm_v0_n_tiling.py
A testing/python/language/test_tilelang_ascend_language_integer_minmax_index.py
A testing/python/language/test_tilelang_ascend_language_l1_to_l0.py
A testing/python/language/test_tilelang_ascend_language_tail_mask_codegen.py
```

## 修改文件

```text
M .agents/skills/tilelang-ascend-tile-api/SKILL.md
M .agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md
M .agents/skills/tilelang-op-design/SKILL.md
M .agents/skills/tilelang-op-design/references/ascend-constraints.md
M .agents/skills/tilelang-op-design/references/quality-checklist.md
M .agents/skills/tilelang-op-generate/references/troubleshooting.md
M .agents/skills/tilelang-pass-analyzer/references/ir-examples.md
M .agents/skills/tilelang-pass-analyzer/references/pass-designs/ascend_infer_buffer_scope_design.md
M .agents/skills/tilelang-pass-analyzer/references/pass-designs/ascend_lower_parallel_to_vector_design.md
M .agents/skills/tilelang-pass-analyzer/references/pass-designs/ascend_memory_planning_technical_doc.md
M .agents/skills/tilelang-pass-analyzer/references/pass-designs/cross_core_pipeline_design.md
M .agents/skills/tilelang-pass-analyzer/references/pass-designs/design_ascend_combinecv.md
M .agents/skills/tilelang-perf-optimization/references/best-practices/cube_optimization_path.md
M .github/workflows/ci_cd.yml
M build_wheel_ascend.sh
M examples/bench_test.sh
M examples/xattention/xattention.py
M examples/xattention/xattention_paged.py
M install_ascend.sh
M pyproject.toml
M setup.py
M src/op/ascend.cc
M src/op/ascend.h
M src/target/codegen_ascend.cc
M src/target/codegen_ascend.h
M src/target/codegen_ascend_pto.cc
M src/target/codegen_ascend_pto.h
M src/tl_templates/ascend/common.h
M src/tl_templates/pto/common.h
M src/transform/allocate_tmp_buffer.cc
M src/transform/ascend_collect_buffer_shape.cc
M src/transform/ascend_combinecv.cc
M src/transform/ascend_infer_buffer_scope.cc
M src/transform/ascend_lower_parallel_to_vector.cc
M src/transform/ascend_memory_planning.cc
M src/transform/ascend_storage_rewrite.cc
M src/transform/ascend_vid_reduction.cc
M src/transform/ascend_workspace_reduction.cc
M src/transform/common/operation_config.h
M src/transform/cross_core_pipeline.cc
M src/transform/legalize_safe_memory_access.cc
M src/transform/lower_tile_op.cc
M src/transform/pipeline_planning.cc
M testing/python/language/cvseparate/test_tilelang_ascend_language_vid_reduction.py
M testing/python/language/test_ascend_memory_planning.py
M testing/python/language/test_tilelang_ascend_language_alloc_codegen.py
M testing/python/language/test_tilelang_ascend_language_elementwise.py
M testing/python/language/test_tilelang_ascend_language_tail_block.py
M tilelang/engine/phase.py
M tilelang/jit/__init__.py
M tilelang/jit/adapter/libgen.py
M tilelang/language/allocate.py
M tilelang/language/copy.py
M tilelang/transform/__init__.py
M tilelang/transform/pass_config.py
```

## 按模块看这次 PR 的主线

### 1. Tail mask / tail block 支持

重点文件：

```text
src/transform/ascend_tail_mask_propagation.cc
src/transform/common/ascend_tail_mask.h
src/tl_templates/ascend/common.h
src/tl_templates/pto/common.h
testing/python/language/test_tilelang_ascend_language_tail_mask_codegen.py
testing/python/language/test_tilelang_ascend_language_tail_block.py
examples/tail_mask/example_tail_add.py
```

这部分是新增和完善 tail block / tail mask 的 IR pass、模板函数和测试。

### 2. sync insert vs pass

重点文件：

```text
src/transform/ascend_sync_insert_vs.cc
testing/python/language/test_ascend_sync_insert_vs.py
tilelang/transform/__init__.py
tilelang/transform/pass_config.py
```

这部分新增一个 `ascend_sync_insert_vs` 相关 pass，并接入 Python transform 包和配置。

### 3. Ascend op / codegen 扩展

重点文件：

```text
src/op/ascend.cc
src/op/ascend.h
src/target/codegen_ascend.cc
src/target/codegen_ascend.h
src/target/codegen_ascend_pto.cc
src/target/codegen_ascend_pto.h
```

这部分涉及 op 注册、AscendC/PTO codegen 打印逻辑、copy / vector / reduce 等相关支持。

### 4. Memory / scope / pipeline 相关 pass 调整

重点文件：

```text
src/transform/allocate_tmp_buffer.cc
src/transform/ascend_infer_buffer_scope.cc
src/transform/ascend_memory_planning.cc
src/transform/ascend_storage_rewrite.cc
src/transform/lower_tile_op.cc
src/transform/pipeline_planning.cc
src/transform/cross_core_pipeline.cc
```

这部分是围绕 buffer scope、临时 buffer、storage rewrite、pipeline 规划等 pass 的适配。

### 5. Python API / JIT / build 接入

重点文件：

```text
tilelang/engine/phase.py
tilelang/jit/__init__.py
tilelang/jit/adapter/libgen.py
tilelang/language/allocate.py
tilelang/language/copy.py
tilelang/transform/__init__.py
tilelang/transform/pass_config.py
pyproject.toml
setup.py
build_wheel_ascend.sh
install_ascend.sh
```

这部分是把 C++ pass/codegen 能力接到 Python 侧、JIT 编译侧和构建流程里。

### 6. 新增测试和例子

重点文件：

```text
testing/python/language/test_ascend_sync_insert_vs.py
testing/python/language/test_tilelang_ascend_language_copy_pad_value.py
testing/python/language/test_tilelang_ascend_language_dynamic_fill.py
testing/python/language/test_tilelang_ascend_language_gemm_v0_n_tiling.py
testing/python/language/test_tilelang_ascend_language_integer_minmax_index.py
testing/python/language/test_tilelang_ascend_language_l1_to_l0.py
testing/python/language/test_tilelang_ascend_language_tail_mask_codegen.py
examples/tail_mask/example_tail_add.py
examples/tile_kernels/mhc/head_compute_mix_kernel.py
examples/tile_kernels/moe/moe_topk_gate.py
```

这些适合用来反推这次 PR 的设计目标和使用方式。

## 建议学习顺序

1. 先看测试：

```text
testing/python/language/test_tilelang_ascend_language_tail_mask_codegen.py
testing/python/language/test_tilelang_ascend_language_tail_block.py
testing/python/language/test_ascend_sync_insert_vs.py
```

2. 再看 pass：

```text
src/transform/ascend_tail_mask_propagation.cc
src/transform/ascend_sync_insert_vs.cc
src/transform/lower_tile_op.cc
src/transform/ascend_memory_planning.cc
```

3. 再看 codegen：

```text
src/target/codegen_ascend.cc
src/target/codegen_ascend_pto.cc
src/tl_templates/ascend/common.h
src/tl_templates/pto/common.h
```

4. 最后看 Python 接入：

```text
tilelang/transform/__init__.py
tilelang/transform/pass_config.py
tilelang/engine/phase.py
tilelang/jit/__init__.py
tilelang/jit/adapter/libgen.py
```

## 常用检查命令

查看 PR 改了哪些文件：

```bash
git diff --name-status origin/ascendc_pto..HEAD
```

查看某个文件的具体 PR 改动：

```bash
git diff origin/ascendc_pto..HEAD -- src/transform/ascend_tail_mask_propagation.cc
```

查看 PR 新增文件：

```bash
git diff --name-status origin/ascendc_pto..HEAD | grep '^A'
```

查看 PR 修改文件：

```bash
git diff --name-status origin/ascendc_pto..HEAD | grep '^M'
```

查看这次 fast-forward 带来的提交：

```bash
git log --oneline origin/ascendc_pto..HEAD
```
