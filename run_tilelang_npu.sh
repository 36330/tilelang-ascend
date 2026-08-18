#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_tilelang_npu.sh /path/to/tilelang/example.py

CANN="${CANN:-/home/developer/Ascend/cann-9.0.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
SCRIPT_PATH="${1:-/mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend/examples/flash_attention/flash_attn_bhsd.py}"

SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "script not found: $SCRIPT_PATH" >&2
  exit 1
fi

if [[ ! -f "$CANN/set_env.sh" ]]; then
  echo "CANN set_env.sh not found: $CANN/set_env.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$CANN/set_env.sh"

TILELANG_WHEEL="${TILELANG_WHEEL:-/home/developer/.local/lib/python3.11/site-packages/tilelang}"

export TVM_LIBRARY_PATH="$TILELANG_WHEEL/lib"
export LD_LIBRARY_PATH="$TILELANG_WHEEL/lib:$CANN/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/developer/.local/lib/python3.11/site-packages:$CANN/python/site-packages:$CANN/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH:-}"
export TILELANG_SCRIPT="$SCRIPT_PATH"
export NPU_DEVICE

# Avoid importing the source-tree tilelang package before the wheel package.
cd /tmp

"$PYTHON_BIN" - <<'PY'
import os
import runpy

import torch
import torch_npu

device_id = int(os.environ.get("NPU_DEVICE", "0"))
torch.npu.set_device(device_id)

# Pre-initialize torch_npu/CANN before TileLang imports its own TVM.
_ = torch.randn((1,), dtype=torch.float16, device="npu")
print(f"torch_npu preinit ok: npu:{device_id}")

runpy.run_path(os.environ["TILELANG_SCRIPT"], run_name="__main__")
PY
