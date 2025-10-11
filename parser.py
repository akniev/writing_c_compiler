from lexer import *
from dataclasses import *

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

@dataclass
class UnaryOperatorNode(AstNode):
    pass

@dataclass
class ComplementNode(UnaryOperatorNode):
    pass

@dataclass
class NegateNode(UnaryOperatorNode):
    pass

@dataclass
class UnaryExpressionNode(ExpressionNode):
    unary_operator: UnaryOperatorNode
    expression: ExpressionNode

def expect(cls: Type, tokens: List["Token"]):
    result = bool(tokens) and isinstance(tokens[0], cls)
    if not result:
        raise SyntaxError("Unexpected symbol")

def take_token(tokens: List["Token"]) -> "Token":
    return tokens.pop(0)

def expect_and_take(cls: Type, tokens: List["Token"]) -> "Token":
    expect(cls, tokens)
    return take_token(tokens)

def is_unary(token):
    return isinstance(token, Tilde) or isinstance(token, Hyphen)

def parse_unary_expression(tokens: List["Token"]) -> UnaryOperatorNode:
    token = take_token(tokens)
    if not is_unary(token):
        raise SyntaxError
    exp = ast_parse_expression(tokens)
    if isinstance(token, Tilde):
        return UnaryExpressionNode(ComplementNode(), exp)
    elif isinstance(token, Hyphen):
        return UnaryExpressionNode(NegateNode(), exp)
    else:
        raise SyntaxError

def ast_parse_expression(tokens: List["Token"]) -> "ExpressionNode":
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
        exp = ast_parse_expression(tokens)
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
    
    s_exp = ast_parse_expression(tokens)

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