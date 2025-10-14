from assembly import *

def reg_to_string_4bytes(reg: "AsmReg") -> str:
    match reg:
        case AsmAX():
            return f"%eax"
        case AsmR10():
            return f"%r10d"
        case AsmR11():
            return f"%r11d"
        case AsmDX():
            return f"%edx"
        case AsmCX():
            return f"%ecx"
        case _:
            raise SyntaxError
        
def reg_to_string_1byte(reg: "AsmReg") -> str:
    match reg:
        case AsmAX():
            return f"%al"
        case AsmR10():
            return f"%r10b"
        case AsmR11():
            return f"%r11b"
        case AsmDX():
            return f"%dl"
        case AsmCX():
            return f"%cl"
        case _:
            raise SyntaxError


def operand_to_string(op: AsmOperand, one_byte = False):
    match op:
        case AsmImmediate(value):
            return f"${value}"
        case AsmRegister(reg):
            return reg_to_string_4bytes(reg) if not one_byte else reg_to_string_1byte(reg)
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


def cond_code_to_suffix(cond_code: AsmCondCode) -> str:
    match cond_code:
        case AsmCondCodeE():
            return "e"
        case AsmCondCodeNE():
            return "ne"
        case AsmCondCodeL():
            return "l"
        case AsmCondCodeLE():
            return "le"
        case AsmCondCodeG():
            return "g"
        case AsmCondCodeGE():
            return "ge"
        case _:
            raise SyntaxError


def print_instruction(ins: AsmInstruction) -> str:
    match ins:
        case AsmMov(src, dst):
            return f"movl {operand_to_string(src)}, {operand_to_string(dst)}"
        case AsmMove8(src, dst):
            return f"movb {operand_to_string(src, one_byte=True)}, {operand_to_string(dst, one_byte=True)}"
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
        case AsmCmp(op1, op2):
            return f"cmpl {operand_to_string(op1)}, {operand_to_string(op2)}"
        case AsmJmp(label):
            return f"jmp .L{label}"
        case AsmJmpCC(cond_code, label):
            return f"j{cond_code_to_suffix(cond_code)} .L{label}"
        case AsmSetCC(cond_code, operand):
            return f"set{cond_code_to_suffix(cond_code)} {operand_to_string(operand, one_byte=True)}"
        case AsmLabel(name):
            return f".L{name}:"
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