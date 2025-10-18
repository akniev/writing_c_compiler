from lexer import *
from dataclasses import *

BINARY_OP_TOKENS = [
    Asterisk, 
    ForwardSlash, 
    PercentSign,
    PlusSign, 
    Hyphen, 
    
    Ampersand, 
    Caret, 
    Pipe, 
    LeftShift, 
    RightShift,
    
    TwoAmbersands,
    TwoPipes,
    
    TwoEqualSigns,
    ExclamationEqualSign,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
    EqualSign,
    
    PlusEqual,
    MinusEqual,
    AsteriskEqual,
    ForwardSlashEqual,
    PercentEqual,

    AmpersandEqual,
    PipeEqual,
    CaretEqual,
    LeftShiftEqual,
    RightShiftEqual,
]

COMPOUND_ASSIGNMENT_TOKENS = [
    PlusEqual,
    MinusEqual,
    AsteriskEqual,
    ForwardSlashEqual,
    PercentEqual,
    AmpersandEqual,
    PipeEqual,
    CaretEqual,
    LeftShiftEqual,
    RightShiftEqual,
]

BINARY_OP_PRECENDENCE_MAP = {
    Asterisk: 50,               # *
    ForwardSlash: 50,           # /
    PercentSign: 50,            # %
    PlusSign: 45,               # +
    Hyphen: 45,                 # -
    LeftShift: 40,              # <<
    RightShift: 40,             # >>
    LessThan: 39,               # <
    GreaterThan: 39,            # >
    LessOrEqual: 39,            # <=
    GreaterOrEqual: 39,         # >=
    TwoEqualSigns: 38,          # ==
    ExclamationEqualSign: 38,   # !=
    Ampersand: 35,              # &
    Caret: 30,                  # ^
    Pipe: 25,                   # |
    TwoAmbersands: 20,          # &&
    TwoPipes: 15,               # ||
    EqualSign: 1,               # =
    PlusEqual: 1,               # +=
    MinusEqual: 1,              # -=
    AsteriskEqual: 1,           # *=
    ForwardSlashEqual: 1,       # /=
    PercentEqual: 1,            # %=
    AmpersandEqual: 1,          # &=
    PipeEqual: 1,               # |=
    CaretEqual: 1,              # ^=
    LeftShiftEqual: 1,          # <<=
    RightShiftEqual: 1,         # >>=
}

def precedence(token: "Token") -> int:
    if type(token) in BINARY_OP_TOKENS:
        return BINARY_OP_PRECENDENCE_MAP[type(token)]
    raise SyntaxError

class AstNode:
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
            if isinstance(f_val, AstNode):
                f_val.pretty_print(indent_str + "  " + f_name + " = ", indent + 2)
            elif isinstance(f_val, list):
                print(indent_str + "  " + f_name + " = [")
                for el in f_val:
                    el.pretty_print(indent = indent + 4)
                
                print(indent_str + "  " + "]")

            else:
                print(indent_str + "  " + f_name + " = " + repr(f_val))

        print(indent_str + ")")


# Top level        

@dataclass
class ProgramNode(AstNode):
    function: "FunctionNode"

@dataclass
class FunctionNode(AstNode):
    name: str
    body: List["BlockItemNode"]

@dataclass
class DeclarationNode(AstNode):
    name: str
    init: Optional["ExpressionNode"]




# Block items

class BlockItemNode(AstNode):
    pass

@dataclass
class StatementBlockItemNode(BlockItemNode):
    statement: "StatementNode"

@dataclass
class DeclarationBlockItemNode(BlockItemNode):
    declaration: "DeclarationNode"




# Statements

class StatementNode(AstNode):
    pass

@dataclass
class ReturnStatementNode(StatementNode):
    exp: "ExpressionNode"

@dataclass
class ExpressionStatementNode(StatementNode):
    exp: "ExpressionNode"

class NullStatementNode(StatementNode):
    pass


# Unary Operators

class UnaryOperatorNode(AstNode):
    pass

class ComplementOperatorNode(UnaryOperatorNode):
    pass

class NegateOperatorNode(UnaryOperatorNode):
    pass

class NotOperatorNode(UnaryOperatorNode):
    pass




# Binary Operators

class BinaryOperatorNode(AstNode):
    pass

class AddOperatorNode(BinaryOperatorNode):
    pass

class SubtractOperatorNode(BinaryOperatorNode):
    pass

class MultiplyOperatorNode(BinaryOperatorNode):
    pass

class DivideOperatorNode(BinaryOperatorNode):
    pass

class RemainderOperatorNode(BinaryOperatorNode):
    pass

class LeftShiftOperatorNode(BinaryOperatorNode):
    pass

class RightShiftOperatorNode(BinaryOperatorNode):
    pass

class BitwiseAndOpeatorNode(BinaryOperatorNode):
    pass

class BitwiseOrOperatorNode(BinaryOperatorNode):
    pass

class BitwiseXorOperatorNode(BinaryOperatorNode):
    pass

class LogicalAndOperatorNode(BinaryOperatorNode):
    pass

class LogicalOrOperatorNode(BinaryOperatorNode):
    pass

class EqualOperatorNode(BinaryOperatorNode):
    pass

class NotEqualOperatorNode(BinaryOperatorNode):
    pass

class LessThanOperatorNode(BinaryOperatorNode):
    pass

class GreaterThanOperatorNode(BinaryOperatorNode):
    pass

class LessOrEqualOperatorNode(BinaryOperatorNode):
    pass

class GreaterOrEqualOperatorNode(BinaryOperatorNode):
    pass

class AssignmentOperatorNode(BinaryOperatorNode):
    pass


# # Compound Assignment Operators

# class PlusCompoundAssignmentOperator(BinaryOperatorNode):
#     pass

# class MinusCompoundAssignmentOperator(BinaryOperatorNode):
#     pass

# class MultiplyCompoundAssignmentOperator(BinaryOperatorNode):
#     pass

# class DivideCompoundAssignmentOperator(BinaryOperatorNode):
#     pass

# class RemainderCompoundAssignmentOperator(BinaryOperatorNode):
#     pass

# class ArithmeticAndEqualCompoundAssignmentOperator(BinaryOperatorNode):
#     pass

# class ArithmeticOrCompoundAssignmentOperator(BinaryOperatorNode):
#     pass

# class ArithmeticXorCompoundAssignmentOperator(BinaryOperatorNode):
#     pass

# class LeftShiftCompoundAssignmentOperator(BinaryOperatorNode):
#     pass

# class RightShiftCompoundAssignmentOperator(BinaryOperatorNode):
#     pass



# Expressions

class ExpressionNode(AstNode):
    pass

@dataclass
class ConstantExpressionNode(ExpressionNode):
    const: int

@dataclass
class UnaryExpressionNode(ExpressionNode):
    unary_operator: "UnaryOperatorNode"
    expression: "ExpressionNode"

@dataclass
class BinaryExpressionNode(ExpressionNode):
    binary_op: "BinaryOperatorNode"
    exp1: "ExpressionNode"
    exp2: "ExpressionNode"

@dataclass
class VariableExpressionNode(ExpressionNode):
    name: str

@dataclass
class AssignmentExpressionNode(ExpressionNode):
    lhs: "ExpressionNode"
    rhs: "ExpressionNode"

@dataclass
class CompoundAssignmentExpressionNode(ExpressionNode):
    binary_op: "BinaryOperatorNode"
    lhs: "ExpressionNode"
    rhs: "ExpressionNode"








def expect(cls: Type, tokens: List["Token"]):
    result = bool(tokens) and isinstance(tokens[0], cls)
    if not result:
        raise SyntaxError("Unexpected symbol")

def take_token(tokens: List["Token"]) -> "Token":
    return tokens.pop(0)

def peek(tokens: List["Token"]) -> Optional["Token"]:
    return tokens[0] if tokens else None

def expect_and_take(cls: Type, tokens: List["Token"]) -> "Token":
    expect(cls, tokens)
    return take_token(tokens)

def is_unary(token):
    return isinstance(token, Tilde) or isinstance(token, Hyphen) or isinstance(token, ExclamationMark)

def get_unary_operator(token):
    match token:
        case Tilde():
            return ComplementOperatorNode()
        case Hyphen():
            return NegateOperatorNode()
        case ExclamationMark():
            return NotOperatorNode()
        case _:
            raise SyntaxError

def ast_parse_unary_expression(tokens: List["Token"]) -> UnaryOperatorNode:
    token = take_token(tokens)
    if not is_unary(token):
        raise SyntaxError
    exp = ast_parse_factor(tokens)
    return UnaryExpressionNode(get_unary_operator(token), exp)

def ast_parse_binop(tokens: List["Token"]) -> BinaryOperatorNode:
    token = take_token(tokens)
    match token:
        case Asterisk():
            return MultiplyOperatorNode()
        case ForwardSlash():
            return DivideOperatorNode()
        case PercentSign():
            return RemainderOperatorNode()
        case PlusSign():
            return AddOperatorNode()
        case Hyphen():
            return SubtractOperatorNode()
        case Ampersand():
            return BitwiseAndOpeatorNode()
        case Caret():
            return BitwiseXorOperatorNode()
        case Pipe():
            return BitwiseOrOperatorNode()
        case LeftShift():
            return LeftShiftOperatorNode()
        case RightShift():
            return RightShiftOperatorNode()
        case TwoAmbersands():
            return LogicalAndOperatorNode()
        case TwoPipes():
            return LogicalOrOperatorNode()
        case TwoEqualSigns():
            return EqualOperatorNode()
        case ExclamationEqualSign():
            return NotEqualOperatorNode()
        case LessThan():
            return LessThanOperatorNode()
        case LessOrEqual():
            return LessOrEqualOperatorNode()
        case GreaterThan():
            return GreaterThanOperatorNode()
        case GreaterOrEqual():
            return GreaterOrEqualOperatorNode()
        case _:
            SyntaxError

def ast_parse_compop(token: Token) -> BinaryOperatorNode:
    match token:
        case PlusEqual():
            return AddOperatorNode()
        case MinusEqual():
            return SubtractOperatorNode()
        case AsteriskEqual():
            return MultiplyOperatorNode()
        case ForwardSlashEqual():
            return DivideOperatorNode()
        case PercentEqual():
            return RemainderOperatorNode()
        case AmpersandEqual():
            return BitwiseAndOpeatorNode()
        case PipeEqual():
            return BitwiseOrOperatorNode()
        case CaretEqual():
            return BitwiseXorOperatorNode()
        case LeftShiftEqual():
            return LeftShiftOperatorNode()
        case RightShiftEqual():
            return RightShiftOperatorNode()
        case _:
            raise SyntaxError("Unknown compound assignment operator!")

def ast_parse_exp(tokens: List["Token"], min_prec) -> "ExpressionNode":
    left = ast_parse_factor(tokens)
    next_token = peek(tokens)
    while type(next_token) in BINARY_OP_TOKENS and precedence(next_token) >= min_prec:
        if isinstance(next_token, EqualSign):
            take_token(tokens)
            right = ast_parse_exp(tokens, precedence(next_token))
            left = AssignmentExpressionNode(left, right)
        elif type(next_token) in COMPOUND_ASSIGNMENT_TOKENS:
            t = take_token(tokens)
            operator = ast_parse_compop(t)
            right = ast_parse_exp(tokens, precedence(next_token))
            left = CompoundAssignmentExpressionNode(operator, left, right)
        else:
            operator = ast_parse_binop(tokens)
            right = ast_parse_exp(tokens, precedence(next_token) + 1)
            left = BinaryExpressionNode(operator, left, right)
        next_token = peek(tokens)
    return left

def ast_parse_factor(tokens: List["Token"]) -> "ExpressionNode":
    token = peek(tokens)
    if token is None:
        raise SyntaxError("Unexpected end of file")

    if isinstance(token, Constant):
        s_const: "Constant" = expect_and_take(Constant, tokens)
        return ConstantExpressionNode(s_const.value)
    elif isinstance(token, Identifier):
        t: Identifier = token
        if t.is_keyword:
            raise SyntaxError
        take_token(tokens)
        return VariableExpressionNode(t.name)
    elif is_unary(token):
        return ast_parse_unary_expression(tokens)
    elif isinstance(token, OpenParenthesis):
        expect_and_take(OpenParenthesis, tokens)
        exp = ast_parse_exp(tokens, 0)
        expect_and_take(CloseParenthesis, tokens)
        return exp
    else:
        raise SyntaxError("Unexpected symbol")


def ast_parse_block_item(tokens: List["Token"]) -> "BlockItemNode":
    match peek(tokens):
        # Declaration
        case Identifier("int", True): 
            return ast_parse_declaration(tokens)
        # Statement
        case _:
            return ast_parse_statement(tokens)

def ast_parse_declaration(tokens: List["Token"]) -> "DeclarationBlockItemNode":
    decl_type: Identifier = expect_and_take(Identifier, tokens)
    if (not decl_type.is_keyword) or (decl_type.name != "int"):
        raise SyntaxError
    decl_identifier: Identifier = expect_and_take(Identifier, tokens)
    t = peek(tokens)
    if isinstance(t, Semicolon):
        decl = DeclarationBlockItemNode(DeclarationNode(decl_identifier.name, None))
        expect_and_take(Semicolon, tokens)
        return decl
    expect_and_take(EqualSign, tokens)
    exp = ast_parse_exp(tokens, 0)
    decl = DeclarationBlockItemNode(DeclarationNode(decl_identifier.name, exp))
    expect_and_take(Semicolon, tokens)
    return decl

def ast_parse_statement(tokens: List["Token"]) -> "StatementBlockItemNode":
    match peek(tokens):
        case Semicolon():
            take_token(tokens)
            return StatementBlockItemNode(NullStatementNode())
        case Identifier("return", True):
            r_statement = ast_parse_return(tokens)
            return StatementBlockItemNode(r_statement)
        case _:
            exp = ast_parse_exp(tokens, 0)
            expect_and_take(Semicolon, tokens)
            return StatementBlockItemNode(ExpressionStatementNode(exp))

def ast_parse_return(tokens: List["Token"]) -> "ReturnStatementNode":
    s_ret: Identifier = expect_and_take(Identifier, tokens)
    if not s_ret.is_keyword:
        raise SyntaxError
    if s_ret.name != "return":
        raise SyntaxError
    
    s_exp = ast_parse_exp(tokens, 0)

    expect_and_take(Semicolon, tokens)

    return ReturnStatementNode(s_exp)


def ast_parse_function(tokens: List["Token"]) -> "FunctionNode":
    expect(Identifier, tokens)
    f_type: "Identifier" = take_token(tokens)
    if not f_type.is_keyword:
        raise SyntaxError
    if f_type.name != "int":
        raise SyntaxError
    
    expect(Identifier, tokens)
    f_name: "Identifier" = take_token(tokens)

    expect_and_take(OpenParenthesis, tokens)

    expect(Identifier, tokens)
    f_args: "Identifier" = take_token(tokens)
    if not f_args.is_keyword:
        raise SyntaxError
    if f_args.name != "void":
        raise SyntaxError
    
    expect_and_take(CloseParenthesis, tokens)

    expect_and_take(OpenBrace, tokens)

    f_statements = []
    
    while not isinstance(peek(tokens), CloseBrace):
        f_statement = ast_parse_block_item(tokens)
        f_statements.append(f_statement)

    expect_and_take(CloseBrace, tokens)

    return FunctionNode(f_name.name, f_statements)


def parse_program(tokens: List["Token"]) -> "ProgramNode":
    func = ast_parse_function(tokens)

    if len(tokens) > 0:
        raise SyntaxError

    p_node = ProgramNode(func)
    return p_node

def parse(tokens: List["Token"]) -> "ProgramNode":
    return parse_program(tokens)