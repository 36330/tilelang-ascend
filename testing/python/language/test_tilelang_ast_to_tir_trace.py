"""Trace how TVMScript expression AST nodes become Python/TIR objects.

This is a learning/debug script, not a normal correctness test.  It copies the
core shape of TVM's ExprEvaluator._visit and prints each important step.

Run from any directory:

    python /mnt/workspace/gitCode/cann/tail-kernel/tilelang-ascend/testing/python/language/test_tilelang_ast_to_tir_trace.py

Useful breakpoint locations:

    TraceExprEvaluator._visit
    TraceExprEvaluator._eval_expr
    TraceExprEvaluator._add_intermediate_result
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict


def _bootstrap_repo_paths() -> Path:
    """Prefer the current checkout's Python code and libtvm over system copies."""

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

import tvm  # noqa: E402
from tvm import tir  # noqa: E402
from tvm.script.parser.core import doc  # noqa: E402
from tvm.script.parser.core.evaluator import DEFAULT_OP  # noqa: E402
import tilelang.language as T  # noqa: E402


def _get_builtin_or_none(name: str):
    builtins = globals().get("__builtins__")
    if not builtins:
        return None
    if not isinstance(builtins, dict) and hasattr(builtins, "__dict__"):
        builtins = builtins.__dict__
    if isinstance(builtins, dict):
        return builtins.get(name)
    return None


def short_obj(value: Any) -> str:
    """Compact one-line object summary for trace logs."""

    if isinstance(value, tir.Buffer):
        shape = [int(x) if isinstance(x, int) else x for x in value.shape]
        return (
            f"tir.Buffer(name={value.name!r}, shape={shape}, "
            f"dtype={value.dtype}, scope={value.scope()})"
        )
    if isinstance(value, tir.Call):
        return f"tir.Call(dtype={value.dtype}, op={value.op})"
    if isinstance(value, tir.PrimExpr):
        return f"{type(value).__name__}({value})"
    if isinstance(value, tvm.ir.Op):
        return f"tvm.ir.Op({value.name})"
    if callable(value):
        module = getattr(value, "__module__", None)
        name = getattr(value, "__name__", type(value).__name__)
        return f"callable({module}.{name})"
    return f"{type(value).__name__}({value!r})"


def dump_ast(node: doc.AST) -> str:
    """Show the equivalent Python ast shape for a doc AST node."""

    py_node = doc.from_doc(node)
    return ast.dump(py_node, include_attributes=False)


class FakeParser:
    """Small stand-in for TVM Parser, only used for error reporting."""

    def report_error(self, node: doc.AST, message: Any) -> None:
        raise RuntimeError(f"{message}\nnode={dump_ast(node)}")


class TraceExprEvaluator:
    """A verbose copy of the core ExprEvaluator._visit flow.

    The point is to see the same important behavior as TVM's real evaluator:

    - Name nodes are returned first, then resolved later.
    - Attribute nodes like T.Tensor are eval'ed and stored as tmp values.
    - Call nodes are rebuilt using those tmp values, then eval'ed again.
    - The outer Call result may be a Buffer, Call, PrimExpr, Python function, etc.
    """

    def __init__(self, value_table: Dict[str, Any]) -> None:
        self.parser = FakeParser()
        self.value_table = value_table
        self.new_value_count = 0
        self.depth = 0

    def log(self, message: str) -> None:
        print(f"{'  ' * self.depth}{message}")

    def _add_intermediate_result(self, value: Any) -> doc.Name:
        name = f"__tvm_tmp_value_{self.new_value_count}"
        self.new_value_count += 1
        self.value_table[name] = value
        self.log(f"store tmp: {name} = {short_obj(value)}")
        return doc.Name(
            id=name,
            ctx=doc.Load(lineno=0, col_offset=0, end_lineno=None, end_col_offset=None),
            lineno=0,
            col_offset=0,
            end_lineno=None,
            end_col_offset=None,
        )

    def _eval_expr(self, node: Any) -> Any:
        """Same idea as tvm.script.parser.core.evaluator._eval_expr."""

        py_node = doc.from_doc(node)
        if isinstance(py_node, ast.expr):
            py_node = ast.Expression(body=py_node)
        py_node = ast.fix_missing_locations(py_node)
        self.log("eval python AST: " + ast.dump(py_node, include_attributes=False))
        exe = compile(py_node, filename="<trace-ast>", mode="eval")
        value = eval(exe, self.value_table)  # pylint: disable=eval-used
        self.log("eval result: " + short_obj(value))
        return value

    def _eval_bin_op(self, fields: Dict[str, Any]) -> Any:
        lhs = self._eval_expr(fields["left"])
        rhs = self._eval_expr(fields["right"])
        op = fields["op"]
        value = DEFAULT_OP[type(op)](lhs, rhs)
        self.log(f"binop result: {short_obj(lhs)} {type(op).__name__} {short_obj(rhs)}")
        self.log("binop value: " + short_obj(value))
        return value

    def _visit(self, node: doc.AST) -> Any:
        if isinstance(node, list):
            self.log(f"visit list[{len(node)}]")
            self.depth += 1
            try:
                return [self._visit(n) for n in node]
            finally:
                self.depth -= 1
        if isinstance(node, tuple):
            self.log(f"visit tuple[{len(node)}]")
            self.depth += 1
            try:
                return tuple(self._visit(n) for n in node)
            finally:
                self.depth -= 1

        assert isinstance(node, doc.AST)
        self.log(f"visit {node.__class__.__name__}: {dump_ast(node)}")

        if isinstance(node, doc.Name):
            if node.id not in self.value_table and not _get_builtin_or_none(node.id):
                raise RuntimeError(f"Undefined variable: {node.id}")
            self.log(f"return unresolved Name({node.id}); real value is resolved by later eval")
            return node

        if isinstance(
            node,
            (
                doc.Constant,
                doc.expr_context,
                doc.operator,
                doc.boolop,
                doc.unaryop,
                doc.cmpop,
            ),
        ):
            self.log(f"return atomic node {node.__class__.__name__}")
            return node

        if not isinstance(node, (doc.expr, doc.slice)):
            self.log(f"return non-expression node {node.__class__.__name__}")
            return node

        fields = {}
        self.depth += 1
        try:
            for field in node.__class__._FIELDS:  # pylint: disable=protected-access
                attr = getattr(node, field)
                if isinstance(attr, (doc.AST, tuple, list)):
                    self.log(f"field {field}: recurse")
                    fields[field] = self._visit(attr)
                else:
                    fields[field] = attr

            if isinstance(node, doc.BinOp):
                value = self._eval_bin_op(fields)
            else:
                rebuilt = node.__class__(**fields)
                self.log("rebuilt node: " + dump_ast(rebuilt))
                value = self._eval_expr(rebuilt)
        except Exception as err:  # pylint: disable=broad-except
            self.parser.report_error(node, err)
        finally:
            self.depth -= 1

        return self._add_intermediate_result(value)

    def eval(self, node: doc.AST) -> Any:
        result = self._visit(node)
        print(f"final _visit returned: {dump_ast(result)}")

        if isinstance(result, doc.Name):
            value = self.value_table[result.id]
            print(f"final resolved value: {short_obj(value)}")
            return value
        if isinstance(result, doc.Constant):
            print(f"final constant value: {result.value!r}")
            return result.value
        raise TypeError(f"Unexpected result type: {type(result)}")


def parse_expr(source: str) -> doc.expr:
    return doc.to_doc(ast.parse(source, mode="eval").body)


def parse_stmt_module(source: str) -> doc.Module:
    return doc.parse(textwrap.dedent(source), mode="exec")


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def trace_expression(source: str, value_table: Dict[str, Any]) -> Any:
    print_section(f"TRACE EXPR: {source}")
    node = parse_expr(source)
    print("python AST:", ast.dump(ast.parse(source, mode="eval"), include_attributes=False))
    print("doc AST   :", dump_ast(node))
    evaluator = TraceExprEvaluator(dict(value_table))
    value = evaluator.eval(node)
    return value


def demo_basic_expression_eval() -> None:
    value_table = {
        "T": T,
        "tir": tir,
        "M": 1024,
        "N": 1024,
    }

    tensor_proxy = trace_expression("T.Tensor", value_table)
    assert tensor_proxy is T.Tensor

    buffer_value = trace_expression('T.Tensor((M, N), "float32")', value_table)
    assert isinstance(buffer_value, tir.Buffer)

    # Prepare real buffers for an op call.  This mimics what function arguments
    # or alloc_buffer variables look like after parser/IRBuilder has created them.
    a_ub = T.Tensor((64, 256), "float32", scope="shared")
    b_ub = T.Tensor((64, 256), "float32", scope="shared")
    value_table.update({"a_ub": a_ub, "b_ub": b_ub})

    square_call = trace_expression("T.square(b_ub, a_ub)", value_table)
    assert isinstance(square_call, tir.Call)

    index_expr = trace_expression("M // 4 * 128 + N // 2", value_table)
    assert isinstance(index_expr, (int, tir.PrimExpr))


def demo_function_ast_shape() -> None:
    print_section("FUNCTION AST SHAPE: find annotation/body Call nodes")

    source = """
    def main(
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
    ):
        a_ub = T.alloc_ub((64, 256), "float32")
        b_ub = T.alloc_ub((64, 256), "float32")
        T.square(b_ub, a_ub)
    """
    module = parse_stmt_module(source)
    func = module.body[0]

    print("FunctionDef name:", func.name)
    for arg in func.args.args:
        print(f"\nargument annotation for {arg.arg}:")
        print("  ", dump_ast(arg.annotation))
        trace_expression(
            ast.unparse(doc.from_doc(arg.annotation)),
            {"T": T, "tir": tir},
        )

    print("\nfunction body AST nodes:")
    for i, stmt in enumerate(func.body):
        print(f"  body[{i}] {stmt.__class__.__name__}: {dump_ast(stmt)}")


def demo_real_prim_func_parse() -> None:
    print_section("REAL @T.prim_func: Python function -> tvm.tir.PrimFunc -> IRModule")

    @T.prim_func
    def main(
        A: T.Tensor((1024, 1024), "float32"),  # type: ignore
        B: T.Tensor((1024, 1024), "float32"),  # type: ignore
    ):
        cid = T.launch_thread("blockIdx.x", 1)
        vid = T.launch_thread("blockIdx.y", 1)
        with T.block("tilelang_root"):
            a_ub = T.alloc_buffer((64, 256), scope="shared")
            b_ub = T.alloc_buffer((64, 256), scope="shared")
            T.reads(A[0, 0], B[0, 0])
            T.writes()
            T.block_attr({"tilelang.is_npu_kernel_frame": T.bool(True)})
            T.evaluate(cid + vid)
            T.square(b_ub, a_ub)

    print("PrimFunc type:", type(main))
    print("PrimFunc attrs:", main.attrs)
    print("PrimFunc params:")
    for param in main.params:
        print("  ", param, param.type_annotation)

    print("\nPrimFunc script:")
    print(main.script())

    mod = tvm.IRModule({"main": main})
    print("\nIRModule type:", type(mod))
    print("IRModule functions:", list(mod.functions.keys()))


def main() -> None:
    print("repo root:", REPO_ROOT)
    print("tvm python:", tvm.__file__)
    print("tilelang language:", T.__file__)
    print("TVM_LIBRARY_PATH:", os.environ.get("TVM_LIBRARY_PATH"))

    demo_basic_expression_eval()
    demo_function_ast_shape()
    demo_real_prim_func_parse()


if __name__ == "__main__":
    main()
