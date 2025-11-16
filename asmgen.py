from assembly import *

def reg_to_string_4bytes(reg: "AsmReg") -> str:
    match reg:
        case AsmAX():
            return f"%eax"
        case AsmCX():
            return f"%ecx"
        case AsmDX():
            return f"%edx"
        case AsmDI():
            return f"%edi"
        case AsmSI():
            return f"%esi"
        case AsmR8():
            return f"%r8d"
        case AsmR9():
            return f"%r9d"
        case AsmR10():
            return f"%r10d"
        case AsmR11():
            return f"%r11d"
        case _:
            raise SyntaxError("Undefined register")
        
def reg_to_string_1byte(reg: "AsmReg") -> str:
    match reg:
        case AsmAX():
            return f"%al"
        case AsmCX():
            return f"%cl"
        case AsmDX():
            return f"%dl"
        case AsmDI():
            return f"%dil"
        case AsmSI():
            return f"%sil"
        case AsmR8():
            return f"%r8b"
        case AsmR9():
            return f"%r9b"
        case AsmR10():
            return f"%r10b"
        case AsmR11():
            return f"%r11b"
        case _:
            raise SyntaxError("Undefined register")

def reg_to_string_8bytes(reg: "AsmReg") -> str:
    match reg:
        case AsmAX():
            return f"%rax"
        case AsmCX():
            return f"%rcx"
        case AsmDX():
            return f"%rdx"
        case AsmDI():
            return f"%rdi"
        case AsmSI():
            return f"%rsi"
        case AsmR8():
            return f"%r8"
        case AsmR9():
            return f"%r9"
        case AsmR10():
            return f"%r10"
        case AsmR11():
            return f"%r11"
        case AsmRSP():
            return f"%rsp"
        case AsmRBP():
            return f"%rbp"
        case _:
            raise SyntaxError("Undefined register")


def operand_to_string(op: AsmOperand, bytes = 4):
    match op:
        case AsmImmediate(value):
            return f"${value}"
        case AsmRegister(reg):
            if bytes == 1:
                return reg_to_string_1byte(reg)
            elif bytes == 4:
                return reg_to_string_4bytes(reg)
            elif bytes == 8:
                return reg_to_string_8bytes(reg)
            else:
                raise SyntaxError
        case AsmStack(value):
            return f"{value}(%rbp)"
        case AsmData(value):
            return f"{value}(%rip)"
        case AsmMem(AsmRegister(reg), offset):
            return f"{offset}({reg_to_string_8bytes(reg)})"
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
            return f"movb {operand_to_string(src, bytes=1)}, {operand_to_string(dst, bytes=1)}"
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
            return f"set{cond_code_to_suffix(cond_code)} {operand_to_string(operand, bytes=1)}"
        case AsmLabel(name):
            return f".L{name}:"
        case AsmRet():
            return f"""
  movq %rbp, %rsp
  popq %rbp
  ret
"""
        case AsmPush(operand):
            return f"pushq {operand_to_string(operand, bytes=8)}"
        case AsmCall(f_name, plt):
            if plt:
                return f"call {f_name}@PLT"
            else:
                return f"call {f_name}"
        case AsmDeallocateStack(value):
            return f"addq ${value}, %rsp"
        case _:
            raise SyntaxError
        

def print_function(f: AsmFunction) -> str:
    asm = ""
    asm += f".text\n"
    if f.is_global:
        asm += f".globl {f.name}\n"
    asm += f"{f.name}:\n"
    asm += f"  pushq %rbp\n"
    asm += f"  movq  %rsp, %rbp\n"

    for ins in f.instructions:
        asm += "  " + print_instruction(ins) + "\n"
    
    return asm

def print_static_variable(s: AsmStaticVariable) -> str:
    asm = ""

    if s.init != 0:
        if s.is_global:
            asm += f"  .globl {s.name}\n"
        asm += f"  .data\n"
        asm += f"  .align 4\n"
        asm += f"{s.name}:\n"
        asm += f"  .long {s.init}\n"
    else:
        if s.is_global:
            asm += f"  .globl {s.name}\n"
        asm += f"  .bss\n"
        asm += f"  .align 4\n"
        asm += f"{s.name}:\n"
        asm += f"  .zero 4\n"
    
    asm += f"\n"

    return asm


def gen_asm(asm_ast: AsmProgram) -> str:
    asm = ""
    functions = []
    static_vars = []

    for entry in asm_ast.top_level_items:
        match entry:
            case AsmFunction(_, _, _) as f:
                functions.append(f)
            case AsmStaticVariable(_, _, _) as s:
                static_vars.append(s)
            case _:
                raise SyntaxError
            
    for f in functions:
        asm += print_function(f)

    for s in static_vars:
        asm += print_static_variable(s)

    asm += '  .section .note.GNU-stack,"",@progbits'

    return asm