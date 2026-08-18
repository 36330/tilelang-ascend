

## 流水图采集
TL_RUN_MODE=sim msprof op simulator \
  --soc-version="Ascend910B3" \
  --kernel-name="main_kernel" \
  --launch-count=1 \
  --output="./opprof_sim_timeline" \
  python /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/softmax/test_case13.py



## details采集
cd /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend/examples/flash_attention

source /home/developer/Ascend/cann-9.0.0/set_env.sh

export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend:$PYTHONPATH

msprof op \
  --aic-metrics="Default,Roofline" \
  --kernel-name="main_kernel" \
  --launch-count=1 \
  --output="./opprof_board" \
  python /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/softmax/test_case13.py


TL_PTO_DEBUG=1 python test_demo.py
  

chmod go-w /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/apply_rotary_pos_emb

msprof op \
  --aic-metrics="Default,Roofline" \
  --kernel-name="kernel_kernel" \
  --launch-count=1 \
  --output="./opprof_board" \
  python /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/apply_rotary_pos_emb/test_demo.py

TL_RUN_MODE=sim msprof op simulator \
  --soc-version="Ascend910B3" \
  --kernel-name="kernel_kernel" \
  --launch-count=1 \
  --output="./opprof_sim_timeline" \
  python /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/apply_rotary_pos_emb/test_demo.py



msprof op \
  --aic-metrics="Default,Roofline" \
  --kernel-name="kernel_kernel" \
  --launch-count=1 \
  --output="./opprof_board" \
  python /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/apply_rotary_pos_emb/test_demo_pipelined.py
  
TL_RUN_MODE=sim msprof op simulator \
  --soc-version="Ascend910B3" \
  --kernel-name="kernel_kernel" \
  --launch-count=1 \
  --output="./opprof_sim_timeline" \
  python /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/apply_rotary_pos_emb/test_demo_pipelined.py







CUSTOM_OP_DIR=/mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/apply_rotary_pos_emb
SOURCE_DIR=$CUSTOM_OP_DIR/cann-bench/source-dir/tilelang_apply_rotary_pos_emb_custom
RESULT_DIR=$CUSTOM_OP_DIR/cann-bench/runs/perf_all_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_DIR"

cd /mnt/workspace/gitCode/cann/tail-kernel/cann-bench

./scripts/run_evaluation.sh \
  --bench-name cann \
  --task-dir tasks/level2/apply_rotary_pos_emb \
  --operator ApplyRotaryPosEmb \
  --source-dir "$SOURCE_DIR" \
  --device-id 0 \
  --reports-dir "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/run.log"



CUSTOM_OP_DIR=/mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/arg_max
SOURCE_DIR=$CUSTOM_OP_DIR/cann-bench/source-dir/tilelang_arg_max_custom
RESULT_DIR=$CUSTOM_OP_DIR/cann-bench/runs/perf_all_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_DIR"

./scripts/run_evaluation.sh \
  --bench-name cann \
  --task-dir tasks/level2/arg_max \
  --operator ArgMax \
  --source-dir "$SOURCE_DIR" \
  --device-id 0 \
  --reports-dir "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/run.log"


CUSTOM_OP_DIR=/mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/gelu
SOURCE_DIR=$CUSTOM_OP_DIR/cann-bench/source-dir/tilelang_gelu_custom
RESULT_DIR=$CUSTOM_OP_DIR/cann-bench/runs/perf_all_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_DIR"

./scripts/run_evaluation.sh \
  --bench-name cann \
  --task-dir tasks/level1/gelu \
  --operator Gelu \
  --source-dir "$SOURCE_DIR" \
  --device-id 0 \
  --reports-dir "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/run.log"


source /home/developer/Ascend/cann-9.0.0/set_env.sh
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/mnt/workspace/gitCode/cann/tail-kernel/cann-bench/src:/mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend:$PYTHONPATH

cd /mnt/workspace/gitCode/cann/tail-kernel/cann-bench

CUSTOM_OP_DIR=/mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/gelu
SOURCE_DIR=$CUSTOM_OP_DIR/cann-bench/source-dir/tilelang_gelu_custom
RESULT_DIR=$CUSTOM_OP_DIR/cann-bench/runs/perf_torch_baseline_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_DIR"

python -m kernel_eval.staged_eval \
  --bench-name cann \
  --task-dir tasks/level1/gelu \
  --operator Gelu \
  --source-dir "$SOURCE_DIR" \
  --device npu \
  --device-id 0 \
  --reports-dir "$RESULT_DIR" \
  --torch-op-guard-mode off \
  2>&1 | tee "$RESULT_DIR/run.log"



## cann-judge

cd /mnt/workspace/gitCode/cann/tail-kernel
source /home/developer/Ascend/cann-9.0.0/set_env.sh
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:$LD_LIBRARY_PATH

bisheng my/hard/hard_swish_manual_db.asc \
  -o my/hard/hard_swish_manual_db \
  --npu-arch=dav-2201

msprof op \
  --aic-metrics="Default,Roofline" \
  --kernel-name="hard_swish_manual_db" \
  --launch-count=1 \
  --output="./my/hard/opprof_manual_db" \
  ./my/hard/hard_swish_manual_db



cd /mnt/workspace/gitCode/cann/tail-kernel
source /home/developer/Ascend/cann-9.0.0/set_env.sh

cmake -S my/hard/manual_db \
  -B my/hard/manual_db/build_sim2 \
  -DCMAKE_ASC_RUN_MODE=sim \
  -DCMAKE_ASC_ARCHITECTURES=dav-2201 \
  -DCMAKE_ASC_FLAGS="-DHARD_SWISH_TOTAL_LENGTH=1048576"

cmake --build my/hard/manual_db/build_sim2 -j

msprof op simulator \
  --soc-version="Ascend910B3" \
  --kernel-name="hard_swish_manual_db" \
  --launch-count=1 \
  --output="./my/hard/opprof_manual_db_sim2" \
  ./my/hard/manual_db/build_sim2/hard_swish_manual_db


## new kernels

chmod go-w /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/gelu

msprof op \
  --aic-metrics="Default,Roofline" \
  --kernel-name="main_kernel" \
  --launch-count=1 \
  --output="./opprof_board" \
  python /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/gelu/cann-bench/source-dir/tilelang_gelu_custom/cann_bench/gelu.py

TL_RUN_MODE=sim msprof op simulator \
  --soc-version="Ascend910B3" \
  --kernel-name="main_kernel" \
  --launch-count=1 \
  --output="./opprof_sim_timeline" \
  python /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/gelu/cann-bench/source-dir/tilelang_gelu_custom/cann_bench/gelu.py


chmod go-w /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/arg_max

msprof op \
  --aic-metrics="Default,Roofline" \
  --kernel-name="main_kernel" \
  --launch-count=1 \
  --output="./opprof_board" \
  python /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/arg_max/cann-bench/source-dir/tilelang_arg_max_custom/cann_bench/arg_max.py

TL_RUN_MODE=sim msprof op simulator \
  --soc-version="Ascend910B3" \
  --kernel-name="main_kernel" \
  --launch-count=1 \
  --output="./opprof_sim_timeline" \
  python /mnt/workspace/gitCode/cann/tail-kernel/cannbot-skills/plugins-official/tilelang-op-orchestrator/custom/arg_max/cann-bench/source-dir/tilelang_arg_max_custom/cann_bench/arg_max.py

## tilelang test

修改后编译安装
cd /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
bash install_ascend.sh --enable-incremental

USE_ASCEND=true ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0 pip install -e .

python /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend/testing/python/language/test_tilelang_ascend_language_my_square.py

python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc --no-source
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh float16 ascendc --no-source
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc --lower-only
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float16 ascendc --lower-only
python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh float16 ascendc --lower-only


## 测试改动
cd /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend/build

PYTHONPATH=/mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend:$PYTHONPATH \
python /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend/testing/python/language/test_tilelang_ascend_language_elementwise_my_abs.py


## tilelang git-push test
cd /mnt/workspace/gitCode/cann/tail-kernel/my/tilelang-ascend
source /home/developer/Ascend/cann-9.0.0/set_env.sh


修改后编译安装

export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
export TVM_LIBRARY_PATH=$PWD/build/tvm
export PYTHONPATH=$PWD:$PWD/3rdparty/tvm/python:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build/tvm:$PWD/build:$ASCEND_HOME_PATH/aarch64-linux/lib64:$LD_LIBRARY_PATH

bash install_ascend.sh --enable-incremental

USE_ASCEND=true ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0 pip install -e .

python /mnt/workspace/gitCode/cann/tail-kernel/my/tilelang-ascend/testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py

python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py --runtime reduce_absmax ascendc
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py --runtime reduce_abssum ascendc
python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py --runtime cumsum ascendc
