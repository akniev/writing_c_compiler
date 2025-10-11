from parser import *
from lexer import *
from dataclasses import *
from tacky import *

class AsmNode:
    pass

@dataclass
class AsmProgram(AsmNode):
    function: "AsmFunction"

@dataclass
class AsmFunction(AsmNode):
    name: str
    instructions: List["AsmInstruction"]

class AsmInstruction:
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

class AsmUnaryOperator(AsmNode):
    pass

class AsmNeg(AsmUnaryOperator):
    pass

class AsmNot(AsmUnaryOperator):
    pass

class AsmOperand:
    pass

@dataclass
class AsmRegister(AsmOperand):
    reg: "AsmReg"

class AsmReg(AsmNode):
    pass

class AsmAX(AsmReg):
    pass

class AsmR10(AsmReg):
    pass

@dataclass
class AsmImmutable(AsmOperand):
    value: int

@dataclass
class AsmPseudo(AsmOperand):
    identifier: str

@dataclass
class AsmStack(AsmOperand):
    value: int

@dataclass
class AsmRet(AsmInstruction):
    pass

def ast_parse_expression(e: ExpressionNode) -> "AsmImmutable":
    return AsmImmutable(e.const)

def ast_parse_return(s: ReturnNode) -> List["AsmInstruction"]:
    src = ast_parse_expression(s.exp)
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

def print_operand(op: AsmOperand):
    match op:
        case AsmRegister():
            return "%eax"
        case AsmImmutable(value):
            return f"${value}"
        case _:
            raise SyntaxError

def print_instruction(ins: AsmInstruction) -> str:
    match ins:
        case AsmMove(src, dst):
            return f"movl {print_operand(src)}, {print_operand(dst)}"
        case AsmRet():
            return "ret"
        case _:
            raise SyntaxError

def print_function(f: AsmFunction) -> str:
    asm = ""
    asm += f".globl {f.name}\n"
    asm += f"{f.name}:\n"

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


def tacky_parse_instruction(t_inst: TInstruction) -> List["AsmInstruction"]:
    match t_inst:
        case TReturnInstruction(val):
            a_val = tacky_parse_value(val),
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


def tacky_parse_program(t_prog: TProgram) -> AsmProgram:
    a_func = tacky_parse_function(t_prog.function)
    return AsmProgram(a_func)