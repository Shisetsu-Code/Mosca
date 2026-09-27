from __future__ import annotations

import ast
import hashlib
import json
import platform
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Hole:
    kind: str


@dataclass(frozen=True)
class Node:
    tag: str
    children: tuple[Any, ...] = ()


PRODUCTIONS: dict[str, tuple[str, ...]] = {
    "stmt_list": ("LIST:CONS", "LIST:END"),
    "stmt": ("STMT:ASSIGN", "STMT:AUGASSIGN", "STMT:FOR", "STMT:IF", "STMT:RETURN"),
    "target": ("TARGET:NAME:acc", "TARGET:NAME:x"),
    "expr": (
        "EXPR:NAME:xs", "EXPR:NAME:x", "EXPR:NAME:acc",
        "EXPR:CONST:-1", "EXPR:CONST:0", "EXPR:CONST:1",
        "EXPR:BINOP", "EXPR:COMPARE", "EXPR:SUBSCRIPT",
    ),
    "binop": ("BINOP:ADD", "BINOP:SUB", "BINOP:MULT", "BINOP:FLOORDIV"),
    "cmpop": ("CMPOP:GT", "CMPOP:GTE", "CMPOP:LT", "CMPOP:LTE", "CMPOP:EQ", "CMPOP:NE"),
}

AST_FIELDS: dict[str, tuple[str, ...]] = {
    "Assign": ("targets", "value", "type_comment"),
    "AugAssign": ("target", "op", "value"),
    "For": ("target", "iter", "body", "orelse", "type_comment"),
    "If": ("test", "body", "orelse"),
    "Return": ("value",),
    "Name": ("id", "ctx"),
    "Constant": ("value", "kind"),
    "BinOp": ("left", "op", "right"),
    "Compare": ("left", "ops", "comparators"),
    "Subscript": ("value", "slice", "ctx"),
}


def initial_tree() -> Node:
    return Node("Function", (Hole("stmt_list"),))


def first_hole(term: Any) -> Hole | None:
    if isinstance(term, Hole):
        return term
    if isinstance(term, Node):
        for child in term.children:
            hit = first_hole(child)
            if hit is not None:
                return hit
    if isinstance(term, tuple):
        for child in term:
            hit = first_hole(child)
            if hit is not None:
                return hit
    return None


def replace_first_hole(term: Any, replacement: Any) -> tuple[Any, bool]:
    if isinstance(term, Hole):
        return replacement, True
    if isinstance(term, Node):
        children = list(term.children)
        for i, child in enumerate(children):
            new_child, changed = replace_first_hole(child, replacement)
            if changed:
                children[i] = new_child
                return Node(term.tag, tuple(children)), True
        return term, False
    if isinstance(term, tuple):
        values = list(term)
        for i, child in enumerate(values):
            new_child, changed = replace_first_hole(child, replacement)
            if changed:
                values[i] = new_child
                return tuple(values), True
    return term, False


def count_holes(term: Any) -> int:
    if isinstance(term, Hole):
        return 1
    if isinstance(term, Node):
        return sum(count_holes(x) for x in term.children)
    if isinstance(term, tuple):
        return sum(count_holes(x) for x in term)
    return 0


def count_nodes(term: Any) -> int:
    if isinstance(term, Node):
        return 1 + sum(count_nodes(x) for x in term.children)
    if isinstance(term, tuple):
        return sum(count_nodes(x) for x in term)
    return 0


def tag_counts(term: Any, out: dict[str, int] | None = None) -> dict[str, int]:
    out = {} if out is None else out
    if isinstance(term, Hole):
        key = f"hole:{term.kind}"
        out[key] = out.get(key, 0) + 1
    elif isinstance(term, Node):
        out[term.tag] = out.get(term.tag, 0) + 1
        for child in term.children:
            tag_counts(child, out)
    elif isinstance(term, tuple):
        for child in term:
            tag_counts(child, out)
    return out


def expand(kind: str, action: str) -> Any:
    if action not in PRODUCTIONS[kind]:
        raise ValueError(f"{action!r} is not valid for hole {kind!r}")
    if action == "LIST:CONS":
        return Node("Seq", (Hole("stmt"), Hole("stmt_list")))
    if action == "LIST:END":
        return Node("Empty")
    if action == "STMT:ASSIGN":
        return Node("Assign", (Hole("target"), Hole("expr")))
    if action == "STMT:AUGASSIGN":
        return Node("AugAssign", (Hole("target"), Hole("binop"), Hole("expr")))
    if action == "STMT:FOR":
        return Node("For", (Hole("target"), Hole("expr"), Hole("stmt_list")))
    if action == "STMT:IF":
        return Node("If", (Hole("expr"), Hole("stmt_list")))
    if action == "STMT:RETURN":
        return Node("Return", (Hole("expr"),))
    if action.startswith("TARGET:NAME:") or action.startswith("EXPR:NAME:"):
        return Node("Name", (action.rsplit(":", 1)[1],))
    if action.startswith("EXPR:CONST:"):
        return Node("Constant", (int(action.rsplit(":", 1)[1]),))
    if action == "EXPR:BINOP":
        return Node("BinOp", (Hole("expr"), Hole("binop"), Hole("expr")))
    if action == "EXPR:COMPARE":
        return Node("Compare", (Hole("expr"), Hole("cmpop"), Hole("expr")))
    if action == "EXPR:SUBSCRIPT":
        return Node("Subscript", (Hole("expr"), Hole("expr")))
    if action.startswith("BINOP:") or action.startswith("CMPOP:"):
        return Node(action.split(":", 1)[1])
    raise AssertionError(action)


def _stmt_list(term: Node) -> list[ast.stmt]:
    if term.tag == "Empty":
        return []
    if term.tag != "Seq":
        raise TypeError(f"expected Seq/Empty, got {term.tag}")
    head, tail = term.children
    return [_stmt(head)] + _stmt_list(tail)


def _target(term: Node) -> ast.expr:
    if term.tag != "Name":
        raise TypeError(f"unsupported target {term.tag}")
    return ast.Name(id=term.children[0], ctx=ast.Store())


def _operator(term: Node) -> ast.operator:
    return {"ADD": ast.Add, "SUB": ast.Sub, "MULT": ast.Mult, "FLOORDIV": ast.FloorDiv}[term.tag]()


def _cmpop(term: Node) -> ast.cmpop:
    return {"GT": ast.Gt, "GTE": ast.GtE, "LT": ast.Lt, "LTE": ast.LtE, "EQ": ast.Eq, "NE": ast.NotEq}[term.tag]()


def _expr(term: Node) -> ast.expr:
    if term.tag == "Name":
        return ast.Name(id=term.children[0], ctx=ast.Load())
    if term.tag == "Constant":
        return ast.Constant(value=term.children[0])
    if term.tag == "BinOp":
        left, op, right = term.children
        return ast.BinOp(left=_expr(left), op=_operator(op), right=_expr(right))
    if term.tag == "Compare":
        left, op, right = term.children
        return ast.Compare(left=_expr(left), ops=[_cmpop(op)], comparators=[_expr(right)])
    if term.tag == "Subscript":
        value, index = term.children
        return ast.Subscript(value=_expr(value), slice=_expr(index), ctx=ast.Load())
    raise TypeError(f"unsupported expression {term.tag}")


def _stmt(term: Node) -> ast.stmt:
    if term.tag == "Assign":
        target, value = term.children
        return ast.Assign(targets=[_target(target)], value=_expr(value), type_comment=None)
    if term.tag == "AugAssign":
        target, op, value = term.children
        return ast.AugAssign(target=_target(target), op=_operator(op), value=_expr(value))
    if term.tag == "For":
        target, iterator, body = term.children
        return ast.For(target=_target(target), iter=_expr(iterator), body=_stmt_list(body), orelse=[], type_comment=None)
    if term.tag == "If":
        test, body = term.children
        return ast.If(test=_expr(test), body=_stmt_list(body), orelse=[])
    if term.tag == "Return":
        return ast.Return(value=_expr(term.children[0]))
    raise TypeError(f"unsupported statement {term.tag}")


def to_ast(term: Node) -> ast.Module:
    if count_holes(term):
        raise ValueError("tree still contains holes")
    if term.tag != "Function":
        raise TypeError("root must be Function")
    body = _stmt_list(term.children[0]) or [ast.Pass()]
    fn = ast.FunctionDef(
        name="solve",
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg="xs")], vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
        ),
        body=body, decorator_list=[], returns=None, type_comment=None,
    )
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


def source(term: Node) -> str:
    return ast.unparse(to_ast(term)) + "\n"


def validate_runtime_schema() -> dict[str, tuple[str, ...]]:
    actual: dict[str, tuple[str, ...]] = {}
    for name, expected in AST_FIELDS.items():
        fields = tuple(getattr(ast, name)._fields)
        actual[name] = fields
        if fields != expected:
            raise RuntimeError(f"AST schema mismatch for {name}: expected={expected}, actual={fields}")
    return actual


def rule_manifest() -> dict[str, Any]:
    fields = validate_runtime_schema()
    data = {
        "runtime": platform.python_version(),
        "implementation": platform.python_implementation(),
        "ast_fields": fields,
        "productions": PRODUCTIONS,
    }
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    data["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    data["production_count"] = sum(len(v) for v in PRODUCTIONS.values())
    return data
