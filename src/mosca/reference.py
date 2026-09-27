from __future__ import annotations

REFERENCE_ACTIONS: dict[str, tuple[str, ...]] = {
    "sum_list": (
        "LIST:CONS", "STMT:ASSIGN", "TARGET:NAME:acc", "EXPR:CONST:0",
        "LIST:CONS", "STMT:FOR", "TARGET:NAME:x", "EXPR:NAME:xs",
        "LIST:CONS", "STMT:AUGASSIGN", "TARGET:NAME:acc", "BINOP:ADD", "EXPR:NAME:x",
        "LIST:END", "LIST:CONS", "STMT:RETURN", "EXPR:NAME:acc", "LIST:END",
    ),
    "count_positive": (
        "LIST:CONS", "STMT:ASSIGN", "TARGET:NAME:acc", "EXPR:CONST:0",
        "LIST:CONS", "STMT:FOR", "TARGET:NAME:x", "EXPR:NAME:xs",
        "LIST:CONS", "STMT:IF", "EXPR:COMPARE", "EXPR:NAME:x", "CMPOP:GT", "EXPR:CONST:0",
        "LIST:CONS", "STMT:AUGASSIGN", "TARGET:NAME:acc", "BINOP:ADD", "EXPR:CONST:1",
        "LIST:END", "LIST:END", "LIST:CONS", "STMT:RETURN", "EXPR:NAME:acc", "LIST:END",
    ),
    "max_list": (
        "LIST:CONS", "STMT:ASSIGN", "TARGET:NAME:acc", "EXPR:SUBSCRIPT", "EXPR:NAME:xs", "EXPR:CONST:0",
        "LIST:CONS", "STMT:FOR", "TARGET:NAME:x", "EXPR:NAME:xs",
        "LIST:CONS", "STMT:IF", "EXPR:COMPARE", "EXPR:NAME:x", "CMPOP:GT", "EXPR:NAME:acc",
        "LIST:CONS", "STMT:ASSIGN", "TARGET:NAME:acc", "EXPR:NAME:x", "LIST:END", "LIST:END",
        "LIST:CONS", "STMT:RETURN", "EXPR:NAME:acc", "LIST:END",
    ),
}


MOTOR_REFERENCE_ACTIONS: dict[str, tuple[str, ...]] = {
    "sum_list": (
        "SET:acc=0", "FOR:x:xs", "AUG:acc+=x", "END", "RETURN:acc",
    ),
    "count_positive": (
        "SET:acc=0", "FOR:x:xs", "WHEN:x>0:INC1", "END", "RETURN:acc",
    ),
    "max_list": (
        "SET:acc=xs[0]", "FOR:x:xs", "WHEN:x>acc:SETX", "END", "RETURN:acc",
    ),
}
