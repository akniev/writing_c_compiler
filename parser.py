from lexer import *
from dataclasses import *

BINARY_OP_TOKENS = [Asterisk, ForwardSlash, PercentSign, PlusSign, Hyphen]

BINARY_OP_PRECENDENCE_MAP = {
    Asterisk: 50,
    ForwardSlash: 50,
    PercentSign: 50,
    PlusSign: 45,
    Hyphen: 45
}

def precedence(token: "Token") -> int:
    if type(token) in BINARY_OP_TOKENS:
        return BINARY_OP_PRECENDENCE_MAP[token.__class__]
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

        

@dataclass
class ProgramNode(AstNode):
    function: "FunctionNode"

@dataclass
class FunctionNode(AstNode):
    name: str
    body: "StatementNode"

class StatementNode(AstNode):
    pass

@dataclass
class ReturnNode(StatementNode):
    exp: "ExpressionNode"

@dataclass
class ExpressionNode(AstNode):
    pass

@dataclass
class ConstantExpressionNode(ExpressionNode):
    const: int

class UnaryOperatorNode(AstNode):
    pass

@dataclass
class ComplementNode(UnaryOperatorNode):
    pass

@dataclass
class NegateNode(UnaryOperatorNode):
    pass


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


@dataclass
class UnaryExpressionNode(ExpressionNode):
    unary_operator: "UnaryOperatorNode"
    expression: "ExpressionNode"


@dataclass
class BinaryExpressionNode(ExpressionNode):
    binary_op: "BinaryOperatorNode"
    exp1: "ExpressionNode"
    exp2: "ExpressionNode"


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
    return isinstance(token, Tilde) or isinstance(token, Hyphen)

def parse_unary_expression(tokens: List["Token"]) -> UnaryOperatorNode:
    token = take_token(tokens)
    if not is_unary(token):
        raise SyntaxError
    exp = ast_parse_factor(tokens)
    if isinstance(token, Tilde):
        return UnaryExpressionNode(ComplementNode(), exp)
    elif isinstance(token, Hyphen):
        return UnaryExpressionNode(NegateNode(), exp)
    else:
        raise SyntaxError

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
        case _:
            SyntaxError

def ast_parse_exp(tokens: List["Token"], min_prec) -> "ExpressionNode":
    left = ast_parse_factor(tokens)
    next_token = peek(tokens)
    while type(next_token) in BINARY_OP_TOKENS and precedence(next_token) >= min_prec:
        operator = ast_parse_binop(tokens)
        right = ast_parse_exp(tokens, precedence(next_token) + 1)
        left = BinaryExpressionNode(operator, left, right)
        next_token = peek(tokens)
    return left

def ast_parse_factor(tokens: List["Token"]) -> "ExpressionNode":
    token = tokens[0] if tokens else None
    if token is None:
        raise SyntaxError("Unexpected end of file")

    if isinstance(token, Constant):
        s_const: "Constant" = expect_and_take(Constant, tokens)
        return ConstantExpressionNode(s_const.value)
    elif is_unary(token):
        return parse_unary_expression(tokens)
    elif isinstance(token, OpenParenthesis):
        expect_and_take(OpenParenthesis, tokens)
        exp = ast_parse_exp(tokens, 0)
        expect_and_take(CloseParenthesis, tokens)
        return exp
    else:
        raise SyntaxError("Unexpected symbol")


def parse_statement(tokens: List["Token"]) -> "StatementNode":
    match tokens[0]:
        case Identifier("return", True):
            return ast_parse_return(tokens)
        case _:
            raise SyntaxError

def ast_parse_return(tokens: List["Token"]) -> "ReturnNode":
    s_ret: Identifier = expect_and_take(Identifier, tokens)
    if not s_ret.is_keyword:
        raise SyntaxError
    if s_ret.name != "return":
        raise SyntaxError
    
    s_exp = ast_parse_exp(tokens, 0)

    expect_and_take(Semicolon, tokens)

    return ReturnNode(s_exp)


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

    f_statement = parse_statement(tokens)

    expect_and_take(CloseBrace, tokens)

    return FunctionNode(f_name.name, f_statement)


def parse_program(tokens: List["Token"]) -> "ProgramNode":
    func = ast_parse_function(tokens)

    if len(tokens) > 0:
        raise SyntaxError

    return ProgramNode(func)

def parse(tokens: List["Token"]) -> "ProgramNode":
    return parse_program(tokens)