"""Direct debug/runtime script for T.tile.erf and T.tile.tanh on Ascend.

This file is intentionally useful when run directly with python. It prints the
generated kernel source, sample values, error statistics and PASS/FAIL status.

Examples:

    python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py
    python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc
    python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc subsection_polynomial
    python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh float ascendc subsection_compensation
    python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py tanh float16 ascendc --no-source
    python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc --lower-only
    python testing/python/language/test_tilelang_ascend_language_erf_tanh_issue.py erf float ascendc subsection_polynomial --gelu-erf-tail --no-source
"""

from __future__ import annotations

import inspect
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable


def _bootstrap_repo_paths() -> Path:
    """Prefer this checkout's Python code and build/libtvm when run directly."""

    repo = Path(__file__).resolve().parents[3]
    tvm_python = repo / "3rdparty" / "tvm" / "python"
    tvm_build = repo / "build" / "tvm"

    os.environ.setdefault("TVM_LIBRARY_PATH", str(tvm_build))
    for path in (repo, tvm_python):
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)

    return repo


REPO_ROOT = _bootstrap_repo_paths()

import tilelang  # noqa: E402
from tilelang import language as T  # noqa: E402
import tvm  # noqa: E402


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

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def line(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def subline(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def source_of(obj) -> str:
    try:
        return inspect.getsourcefile(obj) or "<unknown>"
    except TypeError:
        return "<unknown>"


def print_api_binding() -> None:
    line("API binding")
    for name in ALL_OPS:
        obj = getattr(T.tile, name, None)
        print(f"T.tile.{name}")
        print(f"  object : {obj}")
        print(f"  module : {getattr(obj, '__module__', '<unknown>')}")
        print(f"  file   : {source_of(obj)}")
        if obj is None:
            continue
        try:
            src = inspect.getsource(obj).strip().splitlines()
            print("  source :")
            for src_line in src[:10]:
                print(f"    {src_line}")
            if len(src) > 10:
                print("    ...")
        except (OSError, TypeError):
            print("  source : <not available>")


def torch_dtype(dtype: str):
    import torch  # pylint: disable=import-outside-toplevel

    if dtype == "float":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype: {dtype}")


def make_kernel(
    op_name: str,
    dtype: str,
    rows: int = 8,
    cols: int = 64,
    algo: str | None = None,
):
    if op_name not in ALL_OPS:
        raise ValueError(f"unknown op: {op_name}")

    @T.prim_func
    def main(
        src: T.Tensor([rows, cols], dtype),  # type: ignore
        out: T.Tensor([rows, cols], dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub([rows, cols], dtype)
            out_ub = T.alloc_ub([rows, cols], dtype)

            if vid == 0:
                T.copy(src, src_ub)
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
                T.copy(out_ub, out)

    return main


def make_input(dtype: str, rows: int = 8, cols: int = 64):
    import torch  # pylint: disable=import-outside-toplevel

    # Deterministic range plus edge values makes it easy to inspect saturation
    # and sign behavior in direct script output.
    data = torch.linspace(-3.0, 3.0, rows * cols, dtype=torch.float32).reshape(rows, cols)
    data[0, 0] = -7.0
    data[0, 1] = -2.5
    data[0, 2] = -1.0
    data[0, 3] = 0.0
    data[0, 4] = 1.0
    data[0, 5] = 2.5
    data[0, 6] = 7.0
    # GELU-erf cancellation point: default PADE may truncate erf(z) to -1.0.
    data[0, 7] = -3.9174
    return data.to(torch_dtype(dtype)).npu()


def reference_result(op_name: str, src):
    import torch  # pylint: disable=import-outside-toplevel

    if op_name == "erf":
        return torch.erf(src)
    if op_name == "tanh":
        return torch.tanh(src)
    raise ValueError(f"unknown op: {op_name}")


def tolerance(dtype: str) -> tuple[float, float]:
    if dtype == "float16":
        return 3e-3, 3e-3
    if dtype == "bfloat16":
        return 2e-2, 2e-2
    return 1e-5, 1e-5



def print_kernel_source(kernel, op_name: str, algo: str | None) -> None:
    subline("Generated kernel source")
    src = kernel.get_kernel_source()
    print(src)

    expected = "AscendC::Erf" if op_name == "erf" else "AscendC::Tanh"
    expected_algo = resolve_algo(op_name, algo)
    print(f"\nsource contains {expected}: {expected in src}")
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
    print(f"requested algo: {format_algo(op_name, algo)}")


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


def print_compare(op_name: str, dtype: str, algo: str | None, src, got, expected) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    got_f32 = got.detach().to(torch.float32)
    expected_f32 = expected.detach().to(torch.float32)
    abs_err = (got_f32 - expected_f32).abs()
    max_abs = abs_err.max().item()
    max_ref = expected_f32.abs().max().item()
    max_rel = (abs_err / expected_f32.abs().clamp_min(1e-12)).max().item()

    print("got sample      :", got.flatten()[:16].detach().cpu())
    print("expected sample :", expected.flatten()[:16].detach().cpu())
    print("abs err sample  :", abs_err.flatten()[:16].detach().cpu())
    print("tail point      :", src[0, 7].detach().cpu())
    print("tail got        :", got[0, 7].detach().cpu())
    print("tail expected   :", expected[0, 7].detach().cpu())
    print("tail abs err    :", abs_err[0, 7].detach().cpu())
    if op_name == "erf":
        print_gelu_tail_from_erf(src, got_f32, expected_f32)
    print(f"max_abs_err     : {max_abs:.8e}")
    print(f"max_rel_err     : {max_rel:.8e}")
    print(f"max_ref_abs     : {max_ref:.8e}")

    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
    print(f"torch.assert_close: PASS (rtol={rtol}, atol={atol})")
    print(f"RESULT: PASS op={op_name}, dtype={dtype}, algo={format_algo(op_name, algo)}")


def lower_case(
    op_name: str,
    dtype: str,
    target: str,
    algo: str | None,
    print_source: bool,
) -> bool:
    line(f"LOWER CASE op={op_name}, dtype={dtype}, target={target}, algo={format_algo(op_name, algo)}")
    tilelang.cache.clear_cache()
    func = make_kernel(op_name, dtype, algo=algo)
    print("PrimFunc:")
    print(func.script())

    try:
        kernel = tilelang.compile(func, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
        if print_source:
            print_kernel_source(kernel, op_name, algo)
        print(
            f"LOWER RESULT: PASS op={op_name}, dtype={dtype}, target={target}, "
            f"algo={format_algo(op_name, algo)}"
        )
        return True
    except Exception as err:  # pylint: disable=broad-except
        print(f"LOWER RESULT: FAIL op={op_name}, dtype={dtype}, target={target}")
        print(f"exception type: {type(err).__name__}")
        print(f"exception msg : {err}")
        print_known_compile_hint(op_name, algo, err)
        traceback.print_exc(file=sys.stdout)
        return False


def runtime_case(
    op_name: str,
    dtype: str,
    target: str,
    algo: str | None,
    print_source: bool,
) -> bool:
    line(f"RUNTIME CASE op={op_name}, dtype={dtype}, target={target}, algo={format_algo(op_name, algo)}")

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        print(f"torch import failed: {err}")
        return False

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("torch.npu is not available in this environment")
        return False

    tilelang.cache.clear_cache()
    func = make_kernel(op_name, dtype, algo=algo)

    try:
        kernel = tilelang.compile(func, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
        if print_source:
            print_kernel_source(kernel, op_name, algo)

        src = make_input(dtype)
        expected = reference_result(op_name, src)
        torch.npu.synchronize()
        got = kernel(src)
        torch.npu.synchronize()

        print("input sample    :", src.flatten()[:16].detach().cpu())
        print_compare(op_name, dtype, algo, src, got, expected)
        return True
    except Exception as err:  # pylint: disable=broad-except
        print(f"RESULT: FAIL op={op_name}, dtype={dtype}, target={target}")
        print(f"exception type: {type(err).__name__}")
        print(f"exception msg : {err}")
        print_known_compile_hint(op_name, algo, err)
        traceback.print_exc(file=sys.stdout)
        return False


def print_gelu_tail_from_erf(src, got_erf_f32, expected_erf_f32) -> bool:
    import torch  # pylint: disable=import-outside-toplevel

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


def gelu_erf_tail_case(
    dtype: str,
    target: str,
    algo: str | None,
    print_source: bool,
) -> bool:
    line(f"GELU ERF TAIL CASE dtype={dtype}, target={target}, algo={format_algo('erf', algo)}")

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        print(f"torch import failed: {err}")
        return False

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("torch.npu is not available in this environment")
        return False

    tilelang.cache.clear_cache()
    func = make_kernel("erf", dtype, algo=algo)

    try:
        kernel = tilelang.compile(func, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
        if print_source:
            print_kernel_source(kernel, "erf", algo)

        src = make_input(dtype)
        expected = reference_result("erf", src)
        torch.npu.synchronize()
        got = kernel(src)
        torch.npu.synchronize()

        got_f32 = got.detach().to(torch.float32)
        expected_f32 = expected.detach().to(torch.float32)
        print("tail input      :", src[0, 7].detach().cpu())
        print("tail got erf    :", got[0, 7].detach().cpu())
        print("tail torch erf  :", expected[0, 7].detach().cpu())
        clipped = print_gelu_tail_from_erf(src, got_f32, expected_f32)

        resolved_algo = resolve_algo("erf", algo)
        if resolved_algo == "subsection_polynomial" and clipped:
            print("RESULT: FAIL subsection_polynomial still clips erf tail to -1.0")
            return False
        if resolved_algo == "pade" and clipped:
            print("RESULT: KNOWN_ISSUE pade clips erf tail to -1.0")
            return True

        print(f"RESULT: PASS gelu-erf-tail dtype={dtype}, algo={format_algo('erf', algo)}")
        return True
    except Exception as err:  # pylint: disable=broad-except
        print(f"RESULT: FAIL gelu-erf-tail dtype={dtype}, target={target}")
        print(f"exception type: {type(err).__name__}")
        print(f"exception msg : {err}")
        print_known_compile_hint("erf", algo, err)
        traceback.print_exc(file=sys.stdout)
        return False


def expand_arg(value: str | None, all_values: Iterable[str]) -> list[str]:
    values = list(all_values)
    if value is None or value == "all":
        return values
    if value not in values:
        raise SystemExit(f"unknown value {value}, expected one of {values} or all")
    return [value]


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


def parse_args() -> tuple[bool, bool, bool, list[str], list[str], list[str], str | None]:
    args = sys.argv[1:]
    lower_only = False
    print_source = True
    gelu_erf_tail = False
    if "--lower-only" in args:
        lower_only = True
        args = [arg for arg in args if arg != "--lower-only"]
    if "--no-source" in args:
        print_source = False
        args = [arg for arg in args if arg != "--no-source"]
    if "--gelu-erf-tail" in args:
        gelu_erf_tail = True
        args = [arg for arg in args if arg != "--gelu-erf-tail"]

    if len(args) > 4:
        print("Usage:")
        print(f"  python {Path(__file__).name}")
        print(f"  python {Path(__file__).name} erf float ascendc")
        print(f"  python {Path(__file__).name} erf float ascendc subsection_polynomial")
        print(f"  python {Path(__file__).name} tanh float ascendc subsection_compensation")
        print(f"  python {Path(__file__).name} tanh float16 ascendc --no-source")
        print(f"  python {Path(__file__).name} erf float ascendc --lower-only")
        print(f"  python {Path(__file__).name} erf float ascendc subsection_polynomial --gelu-erf-tail --no-source")
        raise SystemExit(2)

    op_arg = args[0] if len(args) >= 1 else ("erf" if gelu_erf_tail else "all")
    dtype_arg = args[1] if len(args) >= 2 else ("float" if gelu_erf_tail else "all")
    target_arg = args[2] if len(args) >= 3 else "ascendc"
    algo_arg = args[3] if len(args) >= 4 else None

    ops = expand_arg(op_arg, ALL_OPS)
    dtypes = expand_arg(dtype_arg, ALL_DTYPES)
    if gelu_erf_tail and ops != ["erf"]:
        raise SystemExit("--gelu-erf-tail only supports op=erf")
    if gelu_erf_tail and dtypes != ["float"]:
        raise SystemExit("--gelu-erf-tail only supports dtype=float")
    if algo_arg is not None and len(ops) != 1:
        raise SystemExit("algo can only be specified when op is erf or tanh, not all")
    for op_name in ops:
        validate_algo(op_name, algo_arg)

    return lower_only, print_source, gelu_erf_tail, ops, dtypes, expand_arg(target_arg, ALL_TARGETS), algo_arg


def main() -> int:
    print("repo root       :", REPO_ROOT)
    print("tvm python      :", tvm.__file__)
    print("tilelang module :", tilelang.__file__)
    print("TVM_LIBRARY_PATH:", os.environ.get("TVM_LIBRARY_PATH"))

    print_api_binding()

    lower_only, print_source, gelu_erf_tail, ops, dtypes, targets, algo = parse_args()
    failed = 0
    for op_name in ops:
        for dtype in dtypes:
            for target in targets:
                if gelu_erf_tail:
                    ok = gelu_erf_tail_case(dtype, target, algo, print_source)
                elif lower_only:
                    ok = lower_case(op_name, dtype, target, algo, print_source)
                else:
                    ok = runtime_case(op_name, dtype, target, algo, print_source)
                failed += 0 if ok else 1

    line("SUMMARY")
    print(f"failed cases: {failed}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
