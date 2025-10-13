from assembly import *

def reg_to_string(reg: "AsmReg") -> str:
    match reg:
        case AsmAX():
            return f"%eax"
        case AsmR10():
            return f"%r10d"
        case AsmR11():
            return f"%r11d"
        case AsmDX():
            return f"%edx"
        case AsmCL():
            return f"%cl"
        case _:
            raise SyntaxError


def operand_to_string(op: AsmOperand):
    match op:
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
        case AsmAddOp():
            return "addl"
        case AsmSubOp():
            return "subl"
        case AsmMultOp():
            return "imull"
        case AsmShlOp():
            return "sall"
        case AsmShrOp():
            return "sarl"
        case AsmAndOp():
            return "andl"
        case AsmXorOp():
            return "xorl"
        case AsmOrOp():
            return "orl"
        case _:
            raise SyntaxError


def print_instruction(ins: AsmInstruction) -> str:
    match ins:
        case AsmMove(src, dst):
            return f"movl {operand_to_string(src)}, {operand_to_string(dst)}"
        case AsmMove8(src, dst):
            return f"movb {operand_to_string(src)}, {operand_to_string(dst)}"
        case AsmAllocateStack(value):
            return f"subq ${value}, %rsp"
        case AsmUnary(operator, operand):
            return f"{operator_to_string(operator)} {operand_to_string(operand)}"
        case AsmBinary(binop, op1, op2):
            return f"{operator_to_string(binop)} {operand_to_string(op1)}, {operand_to_string(op2)}"
        case AsmIDiv(op):
            return f"idivl {operand_to_string(op)}"
        case AsmCdq():
            return "cdq"
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