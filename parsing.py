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
    body: "ReturnNode"

@dataclass
class ReturnNode(AstNode):
    exp: "ExpressionNode"

@dataclass
class ExpressionNode(AstNode):
    const: int

def expect(cls: Type, tokens: List["Token"]):
    return bool(tokens) and isinstance(tokens[0], cls)

def take_token(tokens: List["Token"]) -> "Token":
    return tokens.pop(0)

def expect_and_take(cls: Type, tokens: List["Token"]) -> "Token":
    expect(cls, tokens)
    return take_token(tokens)

def parse_expression(tokens: List["Token"]) -> "ExpressionNode":
    s_const: "Constant" = expect_and_take(Constant, tokens)
    return ExpressionNode(s_const.value)

def parse_return(tokens: List["Token"]) -> "ReturnNode":
    s_ret: Identifier = expect_and_take(Identifier, tokens)
    if not s_ret.is_keyword:
        raise SyntaxError
    if s_ret.name != "return":
        raise SyntaxError
    
    s_exp = parse_expression(tokens)

    expect_and_take(Semicolon, tokens)

    return ReturnNode(s_exp)


def parse_function(tokens: List["Token"]) -> "FunctionNode":
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

    f_statement = parse_return(tokens)

    expect_and_take(CloseBrace, tokens)

    return FunctionNode(f_name.name, f_statement)


def parse_program(tokens: List["Token"]) -> "ProgramNode":
    func = parse_function(tokens)

    if len(tokens) > 0:
        raise SyntaxError

    return ProgramNode(func)

def parse(tokens: List["Token"]) -> "ProgramNode":
    return parse_program(tokens)