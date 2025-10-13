from dataclasses import *
from typing import List
from parser import *

class TackyNode:
    def pretty_print(self, prefix = "", indent = 0):
        indent_str = " " * indent
        name = self.__class__.__name__
        fs = vars(self)

        if len(fs) == 0:
            if len(prefix) > 0:
                print(prefix + name + "()")
            else:
                print(indent_str + name + "()")
            return
        
        if len(prefix) > 0:
            print(prefix + name + "(")
        else:
            print(indent_str + name + "(")

        for f_name in fs.keys():
            f_val = fs[f_name]
            if isinstance(f_val, TackyNode):
                f_val.pretty_print(indent_str + "  " + f_name + " = ", indent + 2)
            elif isinstance(f_val, list):
                print(indent_str + "  " + f_name + " = [")
                for el in f_val:
                    el.pretty_print(indent = indent + 4)
                
                print(indent_str + "  " + "]")

            else:
                print(indent_str + "  " + f_name + " = " + repr(f_val))

        print(indent_str + ")")


var_counter = 0


@dataclass
class TProgram(TackyNode):
    function: "TFunction"


@dataclass
class TFunction(TackyNode):
    identifier: str
    instructions: List["TInstruction"]



# TACKY Instructions

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


@dataclass
class TBinaryInstruction(TInstruction):
    binaryOp: "TBinaryOperator"
    src1: "TValue"
    src2: "TValue"
    dst: "TValue"





# Values

class TValue(TackyNode):
    pass


@dataclass
class TConstant(TValue):
    value: int


@dataclass
class TVariable(TValue):
    identifier: str




# Binary operators

class TBinaryOperator(TackyNode):
    pass

class TAdditionOperator(TBinaryOperator):
    pass

class TSubtractionOperator(TBinaryOperator):
    pass

class TMultiplicationOperator(TBinaryOperator):
    pass

class TDivisionOperator(TBinaryOperator):
    pass

class TRemainderOperator(TBinaryOperator):
    pass

class TLeftShiftOperator(TBinaryOperator):
    pass

class TRightShiftOperator(TBinaryOperator):
    pass

class TBitwiseAndOperator(TBinaryOperator):
    pass

class TBitwiseXorOperator(TBinaryOperator):
    pass

class TBitwiseOrOperator(TBinaryOperator):
    pass





# Unary Operators

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


def t_parse_binop(op: BinaryOperatorNode) -> TBinaryOperator:
    match op:
        case MultiplyOperatorNode():
            return TMultiplicationOperator()
        case DivideOperatorNode():
            return TDivisionOperator()
        case RemainderOperatorNode():
            return TRemainderOperator()
        case AddOperatorNode():
            return TAdditionOperator()
        case SubtractOperatorNode():
            return TSubtractionOperator()
        case LeftShiftOperatorNode():
            return TLeftShiftOperator()
        case RightShiftOperatorNode():
            return TRightShiftOperator()
        case BitwiseAndOpeatorNode():
            return TBitwiseAndOperator()
        case BitwiseXorOperatorNode():
            return TBitwiseXorOperator()
        case BitwiseOrOperatorNode():
            return TBitwiseOrOperator()
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
        case BinaryExpressionNode(op, exp1, exp2):
            v1 = t_parse_expression(exp1, instructions)
            v2 = t_parse_expression(exp2, instructions)
            dst_name = get_temp_var_name()
            dst = TVariable(dst_name)
            t_op = t_parse_binop(op)
            instructions.append(TBinaryInstruction(t_op, v1, v2, dst))
            return dst
        case _:
            raise SyntaxError