from parser import *
from lexer import *
from dataclasses import *
from tacky import *
import weakref

class AsmNode:
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
            if isinstance(f_val, AsmNode):
                f_val.pretty_print(indent_str + "  " + f_name + " = ", indent + 2)
            elif isinstance(f_val, list):
                print(indent_str + "  " + f_name + " = [")
                for el in f_val:
                    el.pretty_print(indent = indent + 4)
                
                print(indent_str + "  " + "]")

            else:
                print(indent_str + "  " + f_name + " = " + repr(f_val))

        print(indent_str + ")")

@dataclass
class AsmProgram(AsmNode):
    function: "AsmFunction"

@dataclass
class AsmFunction(AsmNode):
    name: str
    instructions: List["AsmInstruction"]


# Instructions

class AsmInstruction(AsmNode):
    pass

@dataclass
class AsmMove(AsmInstruction):
    src: "AsmOperand"
    dst: "AsmOperand"

@dataclass
class AsmUnary(AsmInstruction):
    unary_operator: "AsmUnaryOperator"
    operand: "AsmOperand"

@dataclass
class AsmAllocateStack(AsmInstruction):
    value: int

@dataclass
class AsmRet(AsmInstruction):
    pass

@dataclass
class AsmBinary(AsmInstruction):
    binop: "AsmBinaryOp"
    operand1: "AsmOperand"
    operand2: "AsmOperand"

@dataclass
class AsmIDiv(AsmInstruction):
    operand: "AsmOperand"

@dataclass
class AsmCdq(AsmInstruction):
    pass




# Unary operators

class AsmUnaryOperator(AsmNode):
    pass

class AsmNeg(AsmUnaryOperator):
    pass

class AsmNot(AsmUnaryOperator):
    pass



# Binary operators

class AsmBinaryOp(AsmNode):
    pass

class AsmAddOp(AsmBinaryOp):
    pass

class AsmSubOp(AsmBinaryOp):
    pass

class AsmMultOp(AsmBinaryOp):
    pass








# Asm Registers

class AsmReg(AsmNode):
    pass

class AsmAX(AsmReg):
    pass

class AsmR10(AsmReg):
    pass

class AsmDX(AsmReg):
    pass

class AsmR11(AsmReg):
    pass


# Asm Operands

class AsmOperand:
    pass

@dataclass
class AsmRegister(AsmOperand):
    reg: "AsmReg"

@dataclass
class AsmImmutable(AsmOperand):
    value: int

@dataclass
class AsmPseudo(AsmOperand):
    identifier: str

@dataclass
class AsmStack(AsmOperand):
    value: int








def ast_parse_factor(e: ExpressionNode) -> "AsmImmutable":
    return AsmImmutable(e.const)

def ast_parse_return(s: ReturnNode) -> List["AsmInstruction"]:
    src = ast_parse_factor(s.exp)
    dst = AsmRegister()
    return [
        AsmMove(src, dst),
        AsmRet()
    ]

def ast_parse_function(f: FunctionNode) -> AsmFunction:
    f_name = f.name
    f_insts = ast_parse_return(f.body)
    return AsmFunction(f_name, f_insts)

def ast_parse_asm(ast: ProgramNode) -> AsmProgram:
    f_fun = ast_parse_function(ast.function)
    return AsmProgram(f_fun)

def reg_to_string(reg: "AsmReg"):
    match reg:
        case AsmAX():
            return f"%eax"
        case AsmR10:
            return f"%r10d"

def operand_to_string(op: AsmOperand):
    match op:
        case AsmRegister():
            return "%eax"
        case AsmImmutable(value):
            return f"${value}"
        case AsmRegister(reg):
            return reg_to_string(reg)
        case AsmStack(value):
            return f"{value}(%rbp)"
        case _:
            raise SyntaxError

def operator_to_string(operator: AsmUnaryOperator) -> str:
    match operator:
        case AsmNeg():
            return "negl"
        case AsmNot():
            return "notl"


def print_instruction(ins: AsmInstruction) -> str:
    match ins:
        case AsmMove(src, dst):
            return f"movl {operand_to_string(src)}, {operand_to_string(dst)}"
        case AsmAllocateStack(value):
            return f"subq ${value}, %rsp"
        case AsmUnary(operator, operand):
            return f"{operator_to_string(operator)} {operand_to_string(operand)}"
        case AsmRet():
            return f"""
  movq %rbp, %rsp
  popq %rbp
  ret
"""
        case _:
            raise SyntaxError

def print_function(f: AsmFunction) -> str:
    asm = ""
    asm += f".globl {f.name}\n"
    asm += f"{f.name}:\n"
    asm += f"  pushq %rbp\n"
    asm += f"  movq  %rsp, %rbp\n"

    for ins in f.instructions:
        asm += "  " + print_instruction(ins) + "\n"
    
    return asm

def gen_asm(asm_ast: AsmProgram) -> str:
    asm = ""

    asm += print_function(asm_ast.function)

    asm += '  .section .not.GNU-stack,"",@progbits'

    return asm


def tacky_parse_value(t_val: TValue) -> AsmOperand:
    match t_val:
        case TConstant(value):
            return AsmImmutable(value)
        case TVariable(identifier):
            return AsmPseudo(identifier)
        case _:
            raise SyntaxError
        

def tacky_parse_unary_operator(t_unop: TUnaryOperator) -> AsmUnaryOperator:
    match t_unop:
        case TNegateOp():
            return AsmNeg()
        case TComplementOp():
            return AsmNot()
        case _:
            raise SyntaxError


def tacky_parse_binary_operator(t_binop: TBinaryOperator) -> AsmBinaryOp:
    match t_binop:
        case TAdditionOperator():
            return AsmAddOp()
        case TSubtractionOperator():
            return AsmSubOp()
        case TMultiplicationOperator():
            return AsmMultOp()
        case TDivisionOperator() | TRemainderOperator():
            return AsmIDiv()
        case _:
            raise SyntaxError


def tacky_parse_instruction(t_inst: TInstruction) -> List["AsmInstruction"]:
    match t_inst:
        case TReturnInstruction(val):
            a_val = tacky_parse_value(val)
            return [
                AsmMove(a_val, AsmRegister(AsmAX())),
                AsmRet(),
            ]
        case TUnaryInstruction(unop, src, dst):
            a_unop = tacky_parse_unary_operator(unop)
            a_src = tacky_parse_value(src)
            a_dst = tacky_parse_value(dst)
            return [
                AsmMove(a_src, a_dst),
                AsmUnary(a_unop, a_dst)
            ]
        case TBinaryInstruction(TAdditionOperator()
                                | TSubtractionOperator() 
                                | TMultiplicationOperator() as binop, src1, src2, dst):
            a_binop = tacky_parse_binary_operator(binop)
            a_src1 = tacky_parse_value(src1)
            a_src2 = tacky_parse_value(src2)
            a_dst = tacky_parse_value(dst)
            return [
                AsmMove(a_src1, a_dst),
                AsmBinary(a_binop, a_src2, a_dst)
            ]
        case TBinaryInstruction(TDivisionOperator(), src1, src2, dst):
            a_src1 = tacky_parse_value(src1)
            a_src2 = tacky_parse_value(src2)
            a_dst = tacky_parse_value(dst)
            return [
                AsmMove(a_src1, AsmRegister(AsmAX())),
                AsmCdq(),
                AsmIDiv(a_src2),
                AsmMove(AsmRegister(AsmAX()), a_dst),
            ]
        case TBinaryInstruction(TRemainderOperator(), src1, src2, dst):
            a_src1 = tacky_parse_value(src1)
            a_src2 = tacky_parse_value(src2)
            a_dst = tacky_parse_value(dst)
            return [
                AsmMove(a_src1, AsmRegister(AsmAX())),
                AsmCdq(),
                AsmIDiv(a_src2),
                AsmMove(AsmRegister(AsmDX()), a_dst),
            ]
        case _:
            raise SyntaxError


def tacky_parse_instructions(t_insts: List[TInstruction]) -> List[AsmInstruction]:
    result = []
    for inst in t_insts:
        a_insts = tacky_parse_instruction(inst)
        result.extend(a_insts)
    return result


def tacky_parse_function(t_func: TFunction) -> AsmFunction:
    a_name = t_func.identifier
    a_inst = tacky_parse_instructions(t_func.instructions)
    return AsmFunction(a_name, a_inst)


def get_stack_offset(identifier: str, offsets: dict) -> int:
    if not (identifier in offsets):
        n = len(offsets)
        offsets[identifier] = - (n + 1) * 4
    return offsets[identifier]        


def tacky_replace_pseudoregisters(node: AsmNode, offsets: dict) -> AsmNode:
    match node:
        case AsmProgram(asm_func):
            return AsmProgram(tacky_replace_pseudoregisters(asm_func, offsets))
        case AsmFunction(name, f_insts):
            var_stack_offsets = dict()
            a_insts = [tacky_replace_pseudoregisters(f_inst, var_stack_offsets) for f_inst in f_insts]
            var_count = len(var_stack_offsets)
            a_stack = AsmAllocateStack(4 * var_count)
            a_insts = [a_stack] + a_insts

            a_insts_fixed = []
            for a_inst in a_insts:
                a_insts_fixed.extend(tacky_fix_movs(a_inst))

            a_func = AsmFunction(name, a_insts_fixed)
            return a_func
        case AsmMove(src, dst):
            return AsmMove(tacky_replace_pseudoregisters(src, offsets), tacky_replace_pseudoregisters(dst, offsets))
        case AsmUnary(unop, operand):
            return AsmUnary(unop, tacky_replace_pseudoregisters(operand, offsets))
        case AsmBinary(binop, op1, op2):
            return AsmBinary(binop, tacky_replace_pseudoregisters(op1, offsets), tacky_replace_pseudoregisters(op2, offsets))
        case AsmIDiv(op):
            return AsmIDiv(tacky_replace_pseudoregisters(op, offsets))
        case AsmRegister(_) | AsmImmutable(_) | AsmCdq() | AsmRet():
            return node
        case AsmPseudo(identifier):
            return AsmStack(get_stack_offset(identifier, offsets))
        case _:
            raise SyntaxError


def tacky_fix_movs(node: AsmNode) -> List["AsmNode"]:
    match node:
        case AsmMove(AsmStack(val1), AsmStack(val2)):
            return [
                AsmMove(AsmStack(val1), AsmRegister(AsmR10())),
                AsmMove(AsmRegister(AsmR10()), AsmStack(val2))
            ]
        case _:
            return [node]


def tacky_parse_program(t_prog: TProgram) -> AsmProgram:
    a_func = tacky_parse_function(t_prog.function)
    a_prog = AsmProgram(a_func)
    a_prog_pseudo_replaced = tacky_replace_pseudoregisters(a_prog, dict())
    return a_prog_pseudo_replaced