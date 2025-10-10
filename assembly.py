from parsing import *
from tokens import *
from dataclasses import *

class AsmNode:
    pass

@dataclass
class ProgramAsmNode(AsmNode):
    function: "FunctionAsmNode"

@dataclass
class FunctionAsmNode(AsmNode):
    name: str
    instructions: List["InstructionAsmNode"]

class InstructionAsmNode:
    pass

@dataclass
class MovAsmNode(InstructionAsmNode):
    src: "OperandAsmNode"
    dst: "OperandAsmNode"

class OperandAsmNode:
    pass

@dataclass
class RegisterAsmNode(OperandAsmNode):
    pass

@dataclass
class ImmutableAsmNode(OperandAsmNode):
    value: int

@dataclass
class RetAsmNode(InstructionAsmNode):
    pass

def parse_expression(e: ExpressionNode) -> "ImmutableAsmNode":
    return ImmutableAsmNode(e.const)

def parse_return(s: ReturnNode) -> List["InstructionAsmNode"]:
    src = parse_expression(s.exp)
    dst = RegisterAsmNode()
    return [
        MovAsmNode(src, dst),
        RetAsmNode()
    ]

def parse_function(f: FunctionNode) -> FunctionAsmNode:
    f_name = f.name
    f_insts = parse_return(f.body)
    return FunctionAsmNode(f_name, f_insts)

def parse_asm(ast: ProgramNode) -> ProgramAsmNode:
    f_fun = parse_function(ast.function)
    return ProgramAsmNode(f_fun)

def print_operand(op: OperandAsmNode):
    match op:
        case RegisterAsmNode():
            return "%eax"
        case ImmutableAsmNode(value):
            return f"${value}"
        case _:
            raise SyntaxError

def print_instruction(ins: InstructionAsmNode) -> str:
    match ins:
        case MovAsmNode(src, dst):
            return f"movl {print_operand(src)}, {print_operand(dst)}"
        case RetAsmNode():
            return "ret"
        case _:
            raise SyntaxError

def print_function(f: FunctionAsmNode) -> str:
    asm = ""
    asm += f".globl {f.name}\n"
    asm += f"{f.name}:\n"

    for ins in f.instructions:
        asm += "  " + print_instruction(ins) + "\n"
    
    return asm

def gen_asm(asm_ast: ProgramAsmNode) -> str:
    asm = ""

    asm += print_function(asm_ast.function)

    asm += '  .section .not.GNU-stack,"",@progbits'

    return asm
    
    