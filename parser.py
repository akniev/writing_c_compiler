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

    QuestionMark,
]

INC_DEC_TOKENS = [
    TwoPlusses,
    TwoMinuses
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
    TwoPlusses: 60,             # ++
    TwoMinuses: 60,             # --
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
    QuestionMark: 10,           # ?
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
    if type(token) in BINARY_OP_TOKENS + INC_DEC_TOKENS:
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
    body: "BlockNode"

@dataclass
class DeclarationNode(AstNode):
    name: str
    init: Optional["ExpressionNode"]




# Block items

@dataclass
class BlockNode(AstNode):
    items: List["BlockItemNode"]

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

@dataclass
class IfStatementNode(StatementNode):
    cond: "ExpressionNode"
    then_st: "StatementNode"
    else_st: Optional["StatementNode"]

@dataclass
class LabeledStatement(StatementNode):
    name: str

@dataclass
class GotoStatement(StatementNode):
    label: str

@dataclass
class CompoundStatement(StatementNode):
    block: "BlockNode"

class NullStatementNode(StatementNode):
    pass

@dataclass
class BreakStatementNode(StatementNode):
    label: str

@dataclass
class ContinueStatementNode(StatementNode):
    label: str

@dataclass
class WhileStatementNode(StatementNode):
    cond: "ExpressionNode"
    body: "StatementNode"
    label: str

@dataclass
class DoWhileStatementNode(StatementNode):
    body: "StatementNode"
    cond: "ExpressionNode"
    label: str

@dataclass
class ForStatementNode(StatementNode):
    init: "ForInitNode"
    condition: Optional["ExpressionNode"]
    post: Optional["ExpressionNode"]
    body: "StatementNode"
    label: str




# ForInit

class ForInitNode(AstNode):
    pass

@dataclass
class ForInitDeclarationNode(ForInitNode):
    declaration: "DeclarationNode"

@dataclass
class ForInitExpressionNode(ForInitNode):
    expression: Optional["ExpressionNode"]


# Unary Operators

class UnaryOperatorNode(AstNode):
    pass

class ComplementOperatorNode(UnaryOperatorNode):
    pass

class NegateOperatorNode(UnaryOperatorNode):
    pass

class NotOperatorNode(UnaryOperatorNode):
    pass


# Prefix/Postfix Operators

class IncrementOperatorNode(UnaryOperatorNode):
    pass

class DecrementOperatorNode(UnaryOperatorNode):
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
class PrefixExpressionNode(ExpressionNode):
    operator: "UnaryOperatorNode"
    expression: "ExpressionNode"

@dataclass
class PostfixExpressionNode(ExpressionNode):
    operator: "UnaryOperatorNode"
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

@dataclass
class ConditionalExpressionNode(ExpressionNode):
    cond: "ExpressionNode"
    true_exp: "ExpressionNode"
    false_exp: "ExpressionNode"








def expect(cls: Type, tokens: List["Token"]):
    result = bool(tokens) and isinstance(tokens[0], cls)
    if not result:
        raise SyntaxError("Unexpected symbol")
    
def expect_and_take_identifier(name: str, is_keyword: bool, tokens: List["Token"]) -> Identifier:
    t: Identifier = expect_and_take(Identifier, tokens)
    if t.name != name or t.is_keyword != is_keyword:
        raise SyntaxError("Wrong identifier!")
    return t

def take_token(tokens: List["Token"]) -> "Token":
    return tokens.pop(0)

def peek(tokens: List["Token"], steps = 1) -> Optional["Token"]:
    return tokens[steps - 1] if len(tokens) >= steps else None

def expect_and_take(cls: Type, tokens: List["Token"]) -> "Token":
    expect(cls, tokens)
    return take_token(tokens)

def is_unary(token):
    return type(token) in [Tilde, Hyphen, ExclamationMark]

def is_prefix_postfix(token):
    return type(token) in [TwoPlusses, TwoMinuses]

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

def ast_parse_unary_expression(tokens: List["Token"], min_prec) -> UnaryOperatorNode:
    token = take_token(tokens)
    if not is_unary(token):
        raise SyntaxError
    exp = ast_parse_exp(tokens, 60)
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

def ast_parse_prefix_postfix_unop(token: "Token") -> "UnaryOperatorNode":
    match token:
        case TwoPlusses():
            return IncrementOperatorNode()
        case TwoMinuses():
            return DecrementOperatorNode()
        case _:
            raise SyntaxError("Unknown postfix operator!")

def ast_parse_exp(tokens: List["Token"], min_prec) -> "ExpressionNode":
    left = ast_parse_factor(tokens, min_prec)
    next_token = peek(tokens)
    while type(next_token) in BINARY_OP_TOKENS + INC_DEC_TOKENS and precedence(next_token) >= min_prec:
        if isinstance(next_token, EqualSign):
            take_token(tokens)
            right = ast_parse_exp(tokens, precedence(next_token))
            left = AssignmentExpressionNode(left, right)
        elif isinstance(next_token, QuestionMark):
            middle = ast_parse_conditional_middle(tokens)
            right = ast_parse_exp(tokens, precedence(next_token))
            left = ConditionalExpressionNode(left, middle, right)
        elif type(next_token) in [TwoPlusses, TwoMinuses]:
            t = take_token(tokens)
            operator = ast_parse_prefix_postfix_unop(t)
            left = PostfixExpressionNode(operator, left)
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

def ast_parse_conditional_middle(tokens: List["Token"]) -> "ExpressionNode":
    expect_and_take(QuestionMark, tokens)
    exp = ast_parse_exp(tokens, 0)
    expect_and_take(Colon, tokens)
    return exp

def ast_parse_factor(tokens: List["Token"], min_prec) -> "ExpressionNode":
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
        return ast_parse_unary_expression(tokens, min_prec)
    elif is_prefix_postfix(token):
        t = take_token(tokens)
        op = ast_parse_prefix_postfix_unop(t)
        exp = ast_parse_exp(tokens, 1000)
        return PrefixExpressionNode(op, exp)
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
    expect_and_take_identifier("int", True, tokens)
    decl_identifier: Identifier = expect_and_take(Identifier, tokens)
    if decl_identifier.is_keyword:
        raise SyntaxError
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


def ast_parse_while_loop(tokens: List["Token"]) -> WhileStatementNode:
    expect_and_take_identifier("while", True, tokens)
    expect_and_take(OpenParenthesis, tokens)
    cond = ast_parse_exp(tokens, 0)
    expect_and_take(CloseParenthesis, tokens)
    st = ast_parse_statement(tokens)
    return WhileStatementNode(cond, st, "")


def ast_parse_do_while_loop(tokens: List["Token"]) -> DoWhileStatementNode:
    expect_and_take_identifier("do", True, tokens)
    st = ast_parse_statement(tokens)
    expect_and_take_identifier("while", True, tokens)
    expect_and_take(OpenParenthesis, tokens)
    cond = ast_parse_exp(tokens, 0)
    expect_and_take(CloseParenthesis, tokens)
    expect_and_take(Semicolon, tokens)
    return DoWhileStatementNode(st, cond, "")


def ast_parse_for_init(tokens: List["Token"]) -> Optional["ForInitDeclarationNode"]:
    match peek(tokens):
        case Semicolon():
            expect_and_take(Semicolon, tokens)
            return None
        case Identifier("int", True):
            return ast_parse_declaration(tokens).declaration
        case _:
            exp = ast_parse_exp(tokens, 0)
            expect_and_take(Semicolon, tokens)
            return exp


def ast_parse_for_loop(tokens: List["Token"]) -> ForStatementNode:
    expect_and_take_identifier("for", True, tokens)
    expect_and_take(OpenParenthesis, tokens)
    for_init = ast_parse_for_init(tokens)
    for_cond = None
    if not isinstance(peek(tokens), Semicolon):
        for_cond = ast_parse_exp(tokens, 0)
    expect_and_take(Semicolon, tokens)
    for_post = None
    if not isinstance(peek(tokens), CloseParenthesis):
        for_post = ast_parse_exp(tokens, 0)
    expect_and_take(CloseParenthesis, tokens)
    for_body = ast_parse_statement(tokens)
    return ForStatementNode(for_init, for_cond, for_post, for_body, "")


def ast_parse_statement(tokens: List["Token"]) -> "StatementBlockItemNode":
    match peek(tokens):
        case Semicolon():
            take_token(tokens)
            return StatementBlockItemNode(NullStatementNode())
        case Identifier("return", True):
            r_statement = ast_parse_return(tokens)
            return StatementBlockItemNode(r_statement)
        case Identifier("goto", True):
            expect_and_take(Identifier, tokens)
            t: Identifier = expect_and_take(Identifier, tokens)
            expect_and_take(Semicolon, tokens)
            return StatementBlockItemNode(GotoStatement(t.name))
        case Identifier("if", True):
            if_statement = ast_parse_if(tokens)
            return StatementBlockItemNode(if_statement)
        case Identifier(name, False) if isinstance(peek(tokens, 2), Colon):
            expect_and_take(Identifier, tokens)
            expect_and_take(Colon, tokens)
            return StatementBlockItemNode(LabeledStatement(name))
        case Identifier("break", True):
            expect_and_take(Identifier, tokens)
            expect_and_take(Semicolon, tokens)
            return StatementBlockItemNode(BreakStatementNode(""))
        case Identifier("continue", True):
            expect_and_take(Identifier, tokens)
            expect_and_take(Semicolon, tokens)
            return StatementBlockItemNode(ContinueStatementNode(""))
        case Identifier("while", True):
            while_loop = ast_parse_while_loop(tokens)
            return StatementBlockItemNode(while_loop)
        case Identifier("do", True):
            do_while_loop = ast_parse_do_while_loop(tokens)
            return StatementBlockItemNode(do_while_loop)
        case Identifier("for", True):
            for_loop = ast_parse_for_loop(tokens)
            return StatementBlockItemNode(for_loop)
        case OpenBrace():
            block = ast_parse_block(tokens)
            return StatementBlockItemNode(CompoundStatement(block))
        case _:
            exp = ast_parse_exp(tokens, 0)
            expect_and_take(Semicolon, tokens)
            return StatementBlockItemNode(ExpressionStatementNode(exp))

def ast_parse_if(tokens: List["Token"]) -> "IfStatementNode":
    if_token: Identifier = expect_and_take(Identifier, tokens)
    if if_token.name != "if" or not if_token.is_keyword:
        raise SyntaxError("Not a correct if statement!")
    expect_and_take(OpenParenthesis, tokens)
    cond_exp = ast_parse_exp(tokens, 0)
    expect_and_take(CloseParenthesis, tokens)
    then_exp = ast_parse_statement(tokens)
    
    t = peek(tokens)
    else_exp = None
    
    if isinstance(t, Identifier) and t.name == "else" and t.is_keyword:
        expect_and_take(Identifier, tokens)
        else_exp = ast_parse_statement(tokens)
    
    return IfStatementNode(cond_exp, then_exp, else_exp)


def ast_parse_return(tokens: List["Token"]) -> "ReturnStatementNode":
    s_ret: Identifier = expect_and_take(Identifier, tokens)
    if not s_ret.is_keyword:
        raise SyntaxError
    if s_ret.name != "return":
        raise SyntaxError
    
    s_exp = ast_parse_exp(tokens, 0)

    expect_and_take(Semicolon, tokens)

    return ReturnStatementNode(s_exp)


def ast_parse_block(tokens: List["Token"]) -> "BlockNode":
    expect_and_take(OpenBrace, tokens)

    block_items = []
    while not isinstance(peek(tokens), CloseBrace):
        b_item = ast_parse_block_item(tokens)
        block_items.append(b_item)

    expect_and_take(CloseBrace, tokens)

    return BlockNode(block_items)


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

    f_body = ast_parse_block(tokens)

    return FunctionNode(f_name.name, f_body)


def parse_program(tokens: List["Token"]) -> "ProgramNode":
    func = ast_parse_function(tokens)

    if len(tokens) > 0:
        raise SyntaxError

    p_node = ProgramNode(func)
    return p_node

def parse(tokens: List["Token"]) -> "ProgramNode":
    return parse_program(tokens)