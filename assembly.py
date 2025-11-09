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
    functions: List["AsmFunction"]

@dataclass
class AsmFunction(AsmNode):
    name: str
    instructions: List["AsmInstruction"]





# Instructions

class AsmInstruction(AsmNode):
    pass

@dataclass
class AsmMov(AsmInstruction):
    src: "AsmOperand"
    dst: "AsmOperand"

@dataclass
class AsmMove8(AsmInstruction):
    src: "AsmOperand"
    dst: "AsmOperand"

@dataclass
class AsmUnary(AsmInstruction):
    unary_operator: "AsmUnaryOperator"
    operand: "AsmOperand"

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

class AsmCdq(AsmInstruction):
    pass

@dataclass
class AsmCmp(AsmInstruction):
    operand1: "AsmOperand"
    operand2: "AsmOperand"

@dataclass
class AsmJmp(AsmInstruction):
    target: str

@dataclass
class AsmJmpCC(AsmInstruction):
    cond_code: "AsmCondCode"
    target: str

@dataclass
class AsmSetCC(AsmInstruction):
    cond_code: "AsmCondCode"
    operand: "AsmOperand"

@dataclass
class AsmLabel(AsmInstruction):
    name: str

@dataclass
class AsmAllocateStack(AsmInstruction):
    value: int

@dataclass
class AsmDeallocateStack(AsmInstruction):
    val: int

@dataclass
class AsmPush(AsmInstruction):
    operand: "AsmOperand"

@dataclass
class AsmCall(AsmInstruction):
    name: str
    plt: bool



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

class AsmShrOp(AsmBinaryOp):
    pass

class AsmShlOp(AsmBinaryOp):
    pass

class AsmAndOp(AsmBinaryOp):
    pass

class AsmXorOp(AsmBinaryOp):
    pass

class AsmOrOp(AsmBinaryOp):
    pass






# Asm Registers

class AsmReg(AsmNode):
    pass

class AsmAX(AsmReg):
    pass

class AsmCX(AsmReg):
    pass

class AsmDX(AsmReg):
    pass

class AsmDI(AsmReg):
    pass

class AsmSI(AsmReg):
    pass

class AsmR8(AsmReg):
    pass

class AsmR9(AsmReg):
    pass

class AsmR10(AsmReg):
    pass

class AsmR11(AsmReg):
    pass

class AsmRSP(AsmReg):
    pass

class AsmRBP(AsmReg):
    pass





# Asm Operands

class AsmOperand(AsmNode):
    pass

@dataclass
class AsmRegister(AsmOperand):
    reg: "AsmReg"

@dataclass
class AsmImmediate(AsmOperand):
    value: int

@dataclass
class AsmPseudo(AsmOperand):
    identifier: str

@dataclass
class AsmStack(AsmOperand):
    value: int

@dataclass
class AsmMem(AsmOperand):
    addr: AsmRegister
    offset: int





# Condition Codes

class AsmCondCode(AsmNode):
    pass

class AsmCondCodeE(AsmCondCode):
    pass

class AsmCondCodeNE(AsmCondCode):
    pass

class AsmCondCodeG(AsmCondCode):
    pass

class AsmCondCodeGE(AsmCondCode):
    pass

class AsmCondCodeL(AsmCondCode):
    pass

class AsmCondCodeLE(AsmCondCode):
    pass




def ast_parse_factor(e: ExpressionNode) -> "AsmImmediate":
    return AsmImmediate(e.const)

def ast_parse_return(s: ReturnStatementNode) -> List["AsmInstruction"]:
    src = ast_parse_factor(s.exp)
    dst = AsmRegister()
    return [
        AsmMov(src, dst),
        AsmRet()
    ]

def ast_parse_function_declaration(f: FunctionDeclarationNode) -> AsmFunction:
    f_name = f.name
    f_insts = ast_parse_return(f.body)
    return AsmFunction(f_name, f_insts)

def ast_parse_asm(ast: ProgramNode) -> AsmProgram:
    f_fun = ast_parse_function_declaration(ast.function)
    return AsmProgram(f_fun)


def tacky_parse_value(t_val: TValue) -> AsmOperand:
    match t_val:
        case TConstant(value):
            return AsmImmediate(value)
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
        case TLeftShiftOperator():
            return AsmShlOp()
        case TRightShiftOperator():
            return AsmShrOp()
        case TBitwiseAndOperator():
            return AsmAndOp()
        case TBitwiseXorOperator():
            return AsmXorOp()
        case TBitwiseOrOperator():
            return AsmOrOp()
        case _:
            raise SyntaxError


def tacky_parse_relop(binop: TBinaryOperator) -> AsmCondCode:
    match binop:
        case TEqualOperator():
            return AsmCondCodeE()
        case TNotEqualOperator():
            return AsmCondCodeNE()
        case TLessOperator():
            return AsmCondCodeL()
        case TLessOrEqualOperator():
            return AsmCondCodeLE()
        case TGreaterOperator():
            return AsmCondCodeG()
        case TGreaterOrEqualOperator():
            return AsmCondCodeGE()
        case _:
            raise SyntaxError


def tacky_parse_instruction(t_inst: TInstruction) -> List["AsmInstruction"]:
    match t_inst:
        case TReturnInstruction(val):
            a_val = tacky_parse_value(val)
            return [
                AsmMov(a_val, AsmRegister(AsmAX())),
                AsmRet(),
            ]
        case TUnaryInstruction(TNotOp(), src, dst):
            a_src = tacky_parse_value(src)
            a_dst = tacky_parse_value(dst)
            return [
                AsmCmp(AsmImmediate(0), a_src),
                AsmMov(AsmImmediate(0), a_dst),
                AsmSetCC(AsmCondCodeE(), a_dst)
            ]
        case TUnaryInstruction(unop, src, dst):
            a_unop = tacky_parse_unary_operator(unop)
            a_src = tacky_parse_value(src)
            a_dst = tacky_parse_value(dst)
            return [
                AsmMov(a_src, a_dst),
                AsmUnary(a_unop, a_dst)
            ]
        case TBinaryInstruction(TEqualOperator()
                                | TNotEqualOperator()
                                | TLessOperator()
                                | TLessOrEqualOperator()
                                | TGreaterOperator()
                                | TGreaterOrEqualOperator() as binop, src1, src2, dst):
            a_src1 = tacky_parse_value(src1)
            a_src2 = tacky_parse_value(src2)
            a_dst = tacky_parse_value(dst)
            a_binop = tacky_parse_relop(binop)
            return [
                AsmCmp(a_src2, a_src1),
                AsmMov(AsmImmediate(0), a_dst),
                AsmSetCC(a_binop, a_dst)
            ]
        case TBinaryInstruction(TAdditionOperator()
                                | TSubtractionOperator() 
                                | TMultiplicationOperator()
                                | TLeftShiftOperator()
                                | TRightShiftOperator()
                                | TBitwiseAndOperator()
                                | TBitwiseXorOperator()
                                | TBitwiseOrOperator() as binop, src1, src2, dst):
            a_relop = tacky_parse_binary_operator(binop)
            a_src1 = tacky_parse_value(src1)
            a_src2 = tacky_parse_value(src2)
            a_dst = tacky_parse_value(dst)
            return [
                AsmMov(a_src1, a_dst),
                AsmBinary(a_relop, a_src2, a_dst)
            ]
        case TBinaryInstruction(TDivisionOperator(), src1, src2, dst):
            a_src1 = tacky_parse_value(src1)
            a_src2 = tacky_parse_value(src2)
            a_dst = tacky_parse_value(dst)
            return [
                AsmMov(a_src1, AsmRegister(AsmAX())),
                AsmCdq(),
                AsmIDiv(a_src2),
                AsmMov(AsmRegister(AsmAX()), a_dst),
            ]
        case TBinaryInstruction(TRemainderOperator(), src1, src2, dst):
            a_src1 = tacky_parse_value(src1)
            a_src2 = tacky_parse_value(src2)
            a_dst = tacky_parse_value(dst)
            return [
                AsmMov(a_src1, AsmRegister(AsmAX())),
                AsmCdq(),
                AsmIDiv(a_src2),
                AsmMov(AsmRegister(AsmDX()), a_dst),
            ]
        case TCopyInstruction(src, dst):
            a_src = tacky_parse_value(src)
            a_dst = tacky_parse_value(dst)
            return [
                AsmMov(a_src, a_dst)
            ]
        case TJumpInstruction(target):
            return [
                AsmJmp(target)
            ]
        case TJumpIfZeroInstruction(cond, target):
            a_cond = tacky_parse_value(cond)
            return [
                AsmCmp(AsmImmediate(0), a_cond),
                AsmJmpCC(AsmCondCodeE(), target)
            ]
        case TJumpIfNotZeroInstruction(cond, target):
            a_cond = tacky_parse_value(cond)
            return [
                AsmCmp(AsmImmediate(0), a_cond),
                AsmJmpCC(AsmCondCodeNE(), target)
            ]
        case TLabelInstruction(name):
            return [
                AsmLabel(name)
            ]
        case TFunctionCallInstruction(name, args, dst, plt):
            instructions = []
            arg_registers = [AsmDI(), AsmSI(), AsmDX(), AsmCX(), AsmR8(), AsmR9()]

            n = len(args)
            n0 = min(n, len(arg_registers))
            
            register_args = []
            stack_args = []
            stack_padding = 0

            for i in range(n0):
                register_args.append(args[i])
            
            for i in range(n0, n):
                stack_args.append(args[i])

            if len(stack_args) % 2 == 1:
                stack_padding = 8
            else:
                stack_padding = 0
            
            if stack_padding != 0:
                instructions.append(AsmAllocateStack(stack_padding))

            # Pass args in registers
            reg_index = 0
            for tacky_arg in register_args:
                r = arg_registers[reg_index]
                assembly_arg = tacky_parse_value(tacky_arg)
                instructions.append(AsmMov(assembly_arg, AsmRegister(r)))
                reg_index += 1

            # Pass args on stack
            for tacky_arg in reversed(stack_args):
                assembly_arg = tacky_parse_value(tacky_arg)
                match assembly_arg:
                    case AsmRegister() | AsmImmediate():
                        instructions.append(AsmPush(assembly_arg))
                    case _:
                        instructions.append(AsmMov(assembly_arg, AsmRegister(AsmAX())))
                        instructions.append(AsmPush(AsmRegister(AsmAX())))

            # Emit call instruction
            instructions.append(AsmCall(name, plt))

            # Adjust stack pointer
            bytes_to_remove = 8 * len(stack_args) + stack_padding
            if bytes_to_remove != 0:
                instructions.append(AsmDeallocateStack(bytes_to_remove))

            # Retrieve return value
            assembly_dst = tacky_parse_value(dst)
            instructions.append(AsmMov(AsmRegister(AsmAX()), assembly_dst))

            return instructions
        case _:
            raise SyntaxError


def tacky_parse_instructions(t_insts: List[TInstruction]) -> List[AsmInstruction]:
    result = []
    for inst in t_insts:
        a_insts = tacky_parse_instruction(inst)
        result.extend(a_insts)
    return result


def tacky_parse_function(t_func: TFunction) -> AsmFunction:
    instructions = []
    arg_registers = [AsmDI(), AsmSI(), AsmDX(), AsmCX(), AsmR8(), AsmR9()]

    n = len(t_func.params)
    n0 = min(n, len(arg_registers))
    
    register_args = []
    stack_args = []

    for i in range(n0):
        register_args.append(t_func.params[i])
    
    for i in range(n0, n):
        stack_args.append(t_func.params[i])

    reg_index = 0
    for param in register_args:
        instructions.append(AsmMov(AsmRegister(arg_registers[reg_index]), AsmPseudo(param)))
        reg_index += 1
    
    offset = 16
    for param in stack_args:
        instructions.append(AsmMov(AsmMem(AsmRegister(AsmRBP()), offset), AsmRegister(AsmR8())))
        instructions.append(AsmMov(AsmRegister(AsmR8()), AsmPseudo(param)))
        offset += 8

    a_name = t_func.identifier
    instructions.extend(tacky_parse_instructions(t_func.instructions))
    return AsmFunction(a_name, instructions)


def get_stack_offset(identifier: str, offsets: dict) -> int:
    if not (identifier in offsets):
        n = len(offsets)
        offsets[identifier] = - (n + 1) * 4
    return offsets[identifier]


def tacky_fix_asm(node: AsmNode, offsets: dict) -> AsmNode:
    match node:
        case AsmProgram(asm_funcs):
            fixed_asm_funcs = []
            for asm_func in asm_funcs:
                fixed_asm_func = tacky_fix_asm(asm_func, offsets)
                fixed_asm_funcs.append(fixed_asm_func)
            return AsmProgram(fixed_asm_funcs)
        case AsmFunction(name, f_insts):
            var_stack_offsets = dict()
            a_insts = [tacky_fix_asm(f_inst, var_stack_offsets) for f_inst in f_insts]
            var_count = len(var_stack_offsets)
            stack_size = 4 * var_count
            stack_size_rounded = ((stack_size + 15) // 16) * 16
            a_stack = AsmAllocateStack(stack_size_rounded)
            a_insts = [a_stack] + a_insts

            a_movs_fixed = []
            for a_inst in a_insts:
                a_movs_fixed.extend(tacky_fix_movs_adds_subs_cmps(a_inst))

            a_divs_fixed = []
            for a_inst in a_movs_fixed:
                a_divs_fixed.extend(tacky_fix_idivs(a_inst))

            a_func = AsmFunction(name, a_divs_fixed)
            return a_func
        case AsmMov(src, dst):
            return AsmMov(tacky_fix_asm(src, offsets), tacky_fix_asm(dst, offsets))
        case AsmSetCC(cond_code, operand):
            return AsmSetCC(cond_code, tacky_fix_asm(operand, offsets))
        case AsmCmp(operand1, operand2):
            return AsmCmp(tacky_fix_asm(operand1, offsets), tacky_fix_asm(operand2, offsets))
        case AsmUnary(unop, operand):
            return AsmUnary(unop, tacky_fix_asm(operand, offsets))
        case AsmBinary(binop, op1, op2):
            return AsmBinary(binop, tacky_fix_asm(op1, offsets), tacky_fix_asm(op2, offsets))
        case AsmIDiv(op):
            return AsmIDiv(tacky_fix_asm(op, offsets))
        case AsmRegister(_) | AsmImmediate(_) | AsmCdq() | AsmRet() | AsmCmp(_, _) | AsmSetCC(_, _) | AsmJmp(_) | AsmJmpCC(_, _) | AsmLabel(_) | AsmAllocateStack(_) | AsmCall(_, _) | AsmDeallocateStack(_) | AsmStack(_) | AsmMem(_, _):
            return node
        case AsmPseudo(identifier):
            return AsmStack(get_stack_offset(identifier, offsets))
        case AsmPush(op):
            return AsmPush(tacky_fix_asm(op, offsets))
        # case AsmStack(_):
        #     pass # stacks are fixed in tacky_fix_stack
        case _:
            raise SyntaxError


def tacky_fix_idivs(node: AsmNode) -> List["AsmNode"]:
    match node:
        case AsmIDiv(AsmImmediate(_) as op):
            return [
                AsmMov(op, AsmRegister(AsmR10())),
                AsmIDiv(AsmRegister(AsmR10())),
            ]
        case _:
            return [node]


def tacky_fix_movs_adds_subs_cmps(node: AsmNode) -> List["AsmNode"]:
    match node:
        case AsmMov(AsmStack(_) as op1, AsmStack(_) as op2):
            return [
                AsmMov(op1, AsmRegister(AsmR10())),
                AsmMov(AsmRegister(AsmR10()), op2)
            ]
        case AsmBinary(AsmAddOp() 
                       | AsmSubOp() 
                       | AsmAndOp() 
                       | AsmXorOp() 
                       | AsmOrOp() as binop, AsmStack(_) as op1, AsmStack(_) as op2):
            return [
                AsmMov(op1, AsmRegister(AsmR10())),
                AsmBinary(binop, AsmRegister(AsmR10()), op2)
            ]
        case AsmBinary(AsmShlOp() | AsmShrOp() as binop, AsmStack(_) as op1, AsmStack(_) as op2):
            return [
                AsmMove8(op1, AsmRegister(AsmCX())),
                AsmBinary(binop, AsmRegister(AsmCX()), op2)
            ]
        case AsmBinary(AsmMultOp(), op1, AsmStack(_) as op2):
            return [
                AsmMov(op2, AsmRegister(AsmR11())),
                AsmBinary(AsmMultOp(), op1, AsmRegister(AsmR11())),
                AsmMov(AsmRegister(AsmR11()), op2)
            ]
        case AsmCmp(AsmStack(_) as op1, AsmStack(_) as op2):
            return [
                AsmMov(op1, AsmRegister(AsmR10())),
                AsmCmp(AsmRegister(AsmR10()), op2)
            ]
        case AsmCmp(op1, AsmImmediate(_) as op2):
            return [
                AsmMov(op2, AsmRegister(AsmR11())),
                AsmCmp(op1, AsmRegister(AsmR11()))
            ]
        case _:
            return [node]


def tacky_parse_program(t_prog: TProgram) -> AsmProgram:
    t_funcs = []

    for f in t_prog.functions:
        t_func = tacky_parse_function(f)
        t_funcs.append(t_func)
    a_prog = AsmProgram(t_funcs)
    a_prog_pseudo_replaced = tacky_fix_asm(a_prog, dict())
    return a_prog_pseudo_replaced