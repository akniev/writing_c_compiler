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
label_counter = 0


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

@dataclass
class TCopyInstruction(TInstruction):
    src: "TValue"
    dst: "TValue"

@dataclass
class TJumpInstruction(TInstruction):
    target: str

@dataclass
class TJumpIfZeroInstruction(TInstruction):
    condition: "TValue"
    target: str

@dataclass
class TJumpIfNotZeroInstruction(TInstruction):
    condition: "TValue"
    target: str

@dataclass
class TLabelInstruction(TInstruction):
    value: str



# Values

class TValue(TackyNode):
    pass


@dataclass
class TConstant(TValue):
    value: int


@dataclass
class TVariable(TValue):
    identifier: str




# Unary Operators

class TUnaryOperator(TackyNode):
    pass

class TComplementOp(TUnaryOperator):
    pass

class TNegateOp(TUnaryOperator):
    pass

class TNotOp(TUnaryOperator):
    pass

class TIncrementOp(TUnaryOperator):
    pass

class TDecrementOp(TUnaryOperator):
    pass




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

class TEqualOperator(TBinaryOperator):
    pass

class TNotEqualOperator(TBinaryOperator):
    pass

class TLessOperator(TBinaryOperator):
    pass

class TLessOrEqualOperator(TBinaryOperator):
    pass

class TGreaterOperator(TBinaryOperator):
    pass

class TGreaterOrEqualOperator(TBinaryOperator):
    pass


def t_parse_program(pnode: ProgramNode) -> TProgram:
    tfunc = t_parse_function(pnode.function)
    return TProgram(tfunc)


def t_parse_function(fnode: FunctionNode) -> TFunction:
    fname = fnode.name
    finstructions = t_parse_block_items(fnode.body.items)
    finstructions.append(TReturnInstruction(TConstant(0)))
    return TFunction(fname, finstructions)

def t_parse_block_item(b_item: "BlockItemNode") -> List["TInstruction"]:
    match b_item:
        case StatementBlockItemNode(s_node):
            return t_parse_statement(s_node)
        case DeclarationBlockItemNode(d_node):
            return t_parse_declaration(d_node)
        case _:
            raise SyntaxError


def t_parse_declaration(d_node: DeclarationNode) -> List["TInstruction"]:
    instructions = []
    if d_node.init is None:
        return []
    t_parse_assignment(d_node.name, d_node.init, instructions)
    return instructions


def t_parse_assignment(vname: str, rhs: "ExpressionNode", instructions: List["TInstruction"]) -> "TValue":
    result = t_parse_expression(rhs, instructions)
    lhs = TVariable(vname)
    instructions.append(TCopyInstruction(result, lhs))
    return result

def t_parse_compound_assignment(
        binop: BinaryOperatorNode, 
        lhs: "ExpressionNode",
        rhs: "ExpressionNode", 
        instructions: List["TInstruction"]) -> "TValue":
    match lhs:
        case VariableExpressionNode(name):
            exp = BinaryExpressionNode(binop, lhs, rhs)
            result = t_parse_expression(exp, instructions)
            t_lhs = TVariable(name)
            instructions.append(TCopyInstruction(result, t_lhs))
            return result
        case _:
            raise SyntaxError("Wrong lvalue type!")

def t_prefix_postfix_op(op: UnaryOperatorNode) -> TUnaryOperator:
    match op:
        case IncrementOperatorNode():
            return TIncrementOp()
        case DecrementOperatorNode():
            return TDecrementOp()
        case _:
            raise SyntaxError("Unknown prefix or postfix operator!")


def t_parse_prefix_exp(op: UnaryOperatorNode, exp: ExpressionNode, instructions: List["TInstruction"]) -> TValue:
    if not isinstance(exp, VariableExpressionNode):
        raise SyntaxError("Wrong operand!")
    binop = None
    if isinstance(op, IncrementOperatorNode):
        binop = AddOperatorNode()
    elif isinstance(op, DecrementOperatorNode):
        binop = SubtractOperatorNode()
    else:
        raise SyntaxError("Wrong operator!")
    exp1 = CompoundAssignmentExpressionNode(binop, exp, ConstantExpressionNode(1))
    result = t_parse_expression(exp1, instructions)
    return result

def t_parse_cond_exp(cond: "ExpressionNode", 
                     true_exp: "ExpressionNode", 
                     false_exp: "ExpressionNode", 
                     instructions: List["TInstruction"]) -> TValue:
    cond_result = t_parse_expression(cond, instructions)
    e2_label = get_label_name("e2_label")
    end_label = get_label_name("end_label")
    result_var_name = get_temp_var_name("result")
    instructions.extend([
        TJumpIfZeroInstruction(cond_result, e2_label)
    ])
    v1_result = t_parse_expression(true_exp, instructions)
    instructions.extend([
        TCopyInstruction(v1_result, TVariable(result_var_name)),
        TJumpInstruction(end_label),
        TLabelInstruction(e2_label),
    ])
    v2_result = t_parse_expression(false_exp, instructions)
    instructions.extend([
        TCopyInstruction(v2_result, TVariable(result_var_name)),
        TLabelInstruction(end_label),
    ])
    return TVariable(result_var_name)

def t_parse_postfix_exp(op: UnaryOperatorNode, exp: ExpressionNode, instructions: List["TInstruction"]) -> TValue:
    if not isinstance(exp, VariableExpressionNode):
        raise SyntaxError("Wrong operand!")
    binop = None
    if isinstance(op, IncrementOperatorNode):
        binop = TAdditionOperator()
    elif isinstance(op, DecrementOperatorNode):
        binop = TSubtractionOperator()
    else:
        raise SyntaxError("Wrong operator!")
    result: TVariable = t_parse_expression(exp, instructions)
    result1 = TVariable(get_temp_var_name("result"))
    instructions.extend([
        TCopyInstruction(result, result1),
        TBinaryInstruction(binop, result, TConstant(1), result),
    ])
    return result1

def t_parse_block_items(b_items: List["BlockItemNode"]) -> List["TInstruction"]:
    instructions = []

    for b_item in b_items:
        insts = t_parse_block_item(b_item)
        instructions.extend(insts)

    return instructions

def t_parse_if(cond: "ExpressionNode", 
               then_exp: "StatementNode", 
               else_exp: Optional["StatementNode"]) -> List["TInstruction"]:
    instructions = []
    cond_result = t_parse_expression(cond, instructions)

    if else_exp is not None:
        else_label = get_label_name("else")
        end_label = get_label_name("end")
        instructions.extend([
            TJumpIfZeroInstruction(cond_result, else_label)
        ])
        instructions.extend(t_parse_block_item(then_exp))
        instructions.extend([
            TJumpInstruction(end_label),
            TLabelInstruction(else_label),
        ])
        instructions.extend(t_parse_block_item(else_exp))
        instructions.extend([
            TLabelInstruction(end_label)
        ])
    else:
        end_label = get_label_name("end")
        instructions.extend([
            TJumpIfZeroInstruction(cond_result, end_label)
        ])
        instructions.extend(t_parse_block_item(then_exp))
        instructions.extend([
            TLabelInstruction(end_label)
        ])
    return instructions

def t_parse_statement(fstatement: StatementNode) -> List["TInstruction"]:
    match fstatement:
        case ReturnStatementNode(exp):
            instructions = []
            var = t_parse_expression(exp, instructions)
            return instructions + [TReturnInstruction(var)]
        case ExpressionStatementNode(exp):
            instructions = []
            var = t_parse_expression(exp, instructions)
            return instructions
        case IfStatementNode(cond, then_st, else_st):
            return t_parse_if(cond, then_st, else_st)
        case GotoStatement(label):
            return [
                TJumpInstruction(label)
            ]
        case LabeledStatement(name):
            return [
                TLabelInstruction(name)
            ]
        case CompoundStatement(BlockNode(items)):
            return t_parse_block_items(items)
        case NullStatementNode():
            return []
        case _:
            raise SyntaxError


def get_temp_var_name(prefix="tmp") -> str:
    global var_counter
    var_counter += 1
    return f"{prefix}.{var_counter}"

def get_label_name(name: str) -> str:
    global label_counter
    label_counter += 1
    return f"{name}.{label_counter}"


def t_parse_unop(op: UnaryOperatorNode) -> TUnaryOperator:
    match op:
        case ComplementOperatorNode():
            return TComplementOp()
        case NegateOperatorNode():
            return TNegateOp()
        case NotOperatorNode():
            return TNotOp()
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
        case EqualOperatorNode():
            return TEqualOperator()
        case NotEqualOperatorNode():
            return TNotEqualOperator()
        case LessThanOperatorNode():
            return TLessOperator()
        case LessOrEqualOperatorNode():
            return TLessOrEqualOperator()
        case GreaterThanOperatorNode():
            return TGreaterOperator()
        case GreaterOrEqualOperatorNode():
            return TGreaterOrEqualOperator()
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
        case BinaryExpressionNode(LogicalAndOperatorNode(), exp1, exp2):
            false_label_name = get_label_name("false")
            end_label_name = get_label_name("end")
            result = TVariable(get_temp_var_name())
            v1 = t_parse_expression(exp1, instructions)
            instructions.extend([
                TJumpIfZeroInstruction(v1, false_label_name),
            ])
            v2 = t_parse_expression(exp2, instructions)
            instructions.extend([
                TJumpIfZeroInstruction(v2, false_label_name),
                TCopyInstruction(TConstant(1), result),
                TJumpInstruction(end_label_name),
                TLabelInstruction(false_label_name),
                TCopyInstruction(TConstant(0), result),
                TLabelInstruction(end_label_name),
            ])
            return result
        case BinaryExpressionNode(LogicalOrOperatorNode(), exp1, exp2):
            true_label_name = get_label_name("true")
            end_label_name = get_label_name("end")
            result = TVariable(get_temp_var_name())
            v1 = t_parse_expression(exp1, instructions)
            instructions.extend([
                TJumpIfNotZeroInstruction(v1, true_label_name),
            ])
            v2 = t_parse_expression(exp2, instructions)
            instructions.extend([
                TJumpIfNotZeroInstruction(v2, true_label_name),
                TCopyInstruction(TConstant(0), result),
                TJumpInstruction(end_label_name),
                TLabelInstruction(true_label_name),
                TCopyInstruction(TConstant(1), result),
                TLabelInstruction(end_label_name),
            ])
            return result
        case BinaryExpressionNode(op, exp1, exp2):
            v1 = t_parse_expression(exp1, instructions)
            v2 = t_parse_expression(exp2, instructions)
            dst_name = get_temp_var_name()
            dst = TVariable(dst_name)
            t_op = t_parse_binop(op)
            instructions.append(TBinaryInstruction(t_op, v1, v2, dst))
            return dst
        case VariableExpressionNode(name):
            return TVariable(name)
        case AssignmentExpressionNode(VariableExpressionNode(name), rhs):
            return t_parse_assignment(name, rhs, instructions)
        case CompoundAssignmentExpressionNode(binop, lhs, rhs):
            return t_parse_compound_assignment(binop, lhs, rhs, instructions)
        case PrefixExpressionNode(op, exp):
            return t_parse_prefix_exp(op, exp, instructions)
        case PostfixExpressionNode(op, exp):
            return t_parse_postfix_exp(op, exp, instructions)
        case ConditionalExpressionNode(cond, true_exp, false_exp):
            return t_parse_cond_exp(cond, true_exp, false_exp, instructions)
        case _:
            raise SyntaxError