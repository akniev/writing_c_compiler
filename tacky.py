from dataclasses import *
from typing import List
from parser import *

class TackyNode:
    pass


var_counter = 0


@dataclass
class TProgram(TackyNode):
    function: "TFunction"


@dataclass
class TFunction(TackyNode):
    identifier: str
    instructions: List["TInstruction"]


class TInstruction(TackyNode):
    pass

@dataclass
class TReturnInstruction(TInstruction):
    val: "TValue"


@dataclass
class TUnaryInstruction(TInstruction):
    unaryOp: "TUnaryOperator"
    src: "TValue"
    dst: "TValue"


class TValue(TackyNode):
    pass


@dataclass
class TConstant(TValue):
    value: int


@dataclass
class TVariable(TValue):
    identifier: str


class TUnaryOperator(TackyNode):
    pass


class TComplementOp(TUnaryOperator):
    pass


class TNegateOp(TUnaryOperator):
    pass


def t_parse_program(pnode: ProgramNode) -> TProgram:
    tfunc = t_parse_function(pnode.function)
    return TProgram(tfunc)


def t_parse_function(fnode: FunctionNode) -> TFunction:
    fname = fnode.name
    finstructions = t_parse_statement(fnode.body)
    return TFunction(fname, finstructions)


def t_parse_statement(fstatement: StatementNode) -> List["TInstruction"]:
    match fstatement:
        case ReturnNode(exp):
            instructions = []
            var = t_parse_expression(exp, instructions)
            return instructions + [TReturnInstruction(var)]
        case _:
            raise SyntaxError


def get_temp_var_name() -> str:
    global var_counter
    var_counter += 1
    return f"tmp.{var_counter}"


def t_parse_unop(op: UnaryOperatorNode) -> TUnaryOperator:
    match op:
        case ComplementNode():
            return TComplementOp()
        case NegateNode():
            return TNegateOp()
        case _:
            raise SyntaxError


def t_parse_expression(exp: ExpressionNode, instructions: List["TInstruction"]) -> "TValue":
    match exp:
        case ConstantExpressionNode(const):
            return TConstant(const)
        case UnaryExpressionNode(op, exp):
            src = t_parse_expression(exp, instructions)
            dst_name = get_temp_var_name()
            dst = TVariable(dst_name)
            t_op = t_parse_unop(op)
            instructions.append(TUnaryInstruction(t_op, src, dst))
            return dst
        case _:
            raise SyntaxError