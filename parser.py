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
                    if isinstance(el, AstNode):
                        el.pretty_print(indent = indent + 4)
                    else:
                        print(" " * (indent + 4) + repr(el))
                
                print(indent_str + "  " + "]")

            else:
                print(indent_str + "  " + f_name + " = " + repr(f_val))

        print(indent_str + ")")


# MARK: - Top level        

@dataclass
class ProgramNode(AstNode):
    declarations: List["DeclarationNode"]






# MARK: - Specifiers

class DeclarationSpecifierNode(AstNode):
    pass





class StorageClassSpecifierNode(DeclarationSpecifierNode):
    pass

class StaticStorageClassNode(StorageClassSpecifierNode):
    pass

class ExternStorageClassNode(StorageClassSpecifierNode):
    pass




class TypeSpecifierNode(DeclarationSpecifierNode):
    pass

class IntTypeNode(TypeSpecifierNode):
    pass

class LongTypeNode(TypeSpecifierNode):
    pass

@dataclass
class FunTypeNode(TypeSpecifierNode):
    params: List["TypeSpecifierNode"]
    ret: "TypeSpecifierNode"

# MARK: - Declarations

class DeclarationNode(AstNode):
    pass

@dataclass
class VariableDeclarationNode(DeclarationNode):
    name: str
    init: Optional["ExpressionNode"]
    var_type: "TypeSpecifierNode"
    storage_class: Optional["StorageClassSpecifierNode"]

@dataclass
class FunctionDeclarationNode(AstNode):
    name: str
    params: List[str]
    body: Optional["BlockNode"]
    fun_type: "TypeSpecifierNode"
    storage_class: Optional["StorageClassSpecifierNode"]


# MARK: - Block items

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




# MARK: - Statements

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
    statement: "StatementNode"

@dataclass
class CaseLabeledStatement(StatementNode):
    val: "ExpressionNode"
    statement: "StatementNode"
    switch_label: str
    label: str

@dataclass
class DefaultLabeledStatement(StatementNode):
    statement: "StatementNode"
    switch_label: str
    label: str

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

@dataclass
class SwitchStatementNode(StatementNode):
    exp: "ExpressionNode"
    body: "StatementNode"
    cases: List[Tuple[int|None, str]]
    defaultCase: str|None
    label: str



# MARK: - ForInit

class ForInitNode(AstNode):
    pass

@dataclass
class ForInitDeclarationNode(ForInitNode):
    declaration: "VariableDeclarationNode"

@dataclass
class ForInitExpressionNode(ForInitNode):
    expression: Optional["ExpressionNode"]


# MARK: - Unary Operators

class UnaryOperatorNode(AstNode):
    pass

class ComplementOperatorNode(UnaryOperatorNode):
    pass

class NegateOperatorNode(UnaryOperatorNode):
    pass

class NotOperatorNode(UnaryOperatorNode):
    pass


# MARK: - Prefix/Postfix Operators

class IncrementOperatorNode(UnaryOperatorNode):
    pass

class DecrementOperatorNode(UnaryOperatorNode):
    pass




# MARK: - Binary Operators

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


# MARK: - Expressions

class ExpressionNode(AstNode):
    pass

@dataclass
class ConstantExpressionNode(ExpressionNode):
    const: "ConstNode"

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
class CastExpressionNode(ExpressionNode):
    target_type: "TypeSpecifierNode"
    exp: "ExpressionNode"

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

@dataclass
class FunctionCallExpressionNode(ExpressionNode):
    name: str
    args: List["ExpressionNode"]
    plt: bool


# MARK: - Constants

class ConstNode:
    pass

@dataclass
class ConstInt(ConstNode):
    value: int

@dataclass
class ConstLong(ConstNode):
    value: int






def expect(cls: Type, tokens: List["Token"]):
    result = bool(tokens) and isinstance(tokens[0], cls)
    if not result:
        raise SyntaxError("Unexpected symbol")
    
def expect_and_take_identifier(name: str, is_keyword: bool, tokens: List["Token"]) -> Identifier:
    t: Identifier = expect_and_take(Identifier, tokens)
    if t.name != name or t.is_keyword != is_keyword:
        raise SyntaxError("Wrong identifier!")
    return t

def check_identifier(name: str, is_keyword: bool, token: Token|None) -> bool:
    if token is None:
        return False
    if not isinstance(token, Identifier):
        return False
    t: Identifier = token
    if t.is_keyword != is_keyword or t.name != name:
        return False
    return True

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

def ast_parse_function_call(tokens: List["Token"]) -> "FunctionCallExpressionNode":
    f_identifier: Identifier = expect_and_take(Identifier, tokens)
    f_name = f_identifier.name
    f_args = []
    expect_and_take(OpenParenthesis, tokens)
    if not isinstance(peek(tokens), CloseParenthesis):
        e = ast_parse_exp(tokens, 0)
        f_args.append(e)
        while not isinstance(peek(tokens), CloseParenthesis):
            expect_and_take(Comma, tokens)
            e = ast_parse_exp(tokens, 0)
            f_args.append(e)
    
    expect_and_take(CloseParenthesis, tokens)

    return FunctionCallExpressionNode(f_name, f_args, False)


def ast_parse_constant(tokens: List["Token"]) -> "ExpressionNode":
    match peek(tokens):
        case Constant(value) | LongIntegerConstant(value):
            if value > 2**63 - 1:
                raise SyntaxError("Constant is too large to be represented by long")
            take_token(tokens)
            if value > 2**31 - 1:
                return ConstantExpressionNode(ConstLong(value))
            else:
                return ConstantExpressionNode(ConstInt(value))
        case _:
            raise SyntaxError("Unknown constant type")
            

def ast_get_type(tokens: List["Token"]) -> "TypeSpecifierNode":
    specifiers = []
    while isinstance(peek(tokens), Identifier) and peek(tokens).name in ["int", "long"]:
        t: Identifier = expect_and_take(Identifier, tokens)
        specifiers.append(t)
    
    return ast_type_from_tokens(specifiers)

def ast_type_from_tokens(tokens: List["Token"]) -> "TypeSpecifierNode":
    specifiers = [token.name for token in tokens]

    if specifiers == ["int"]:
        return IntTypeNode()
    
    if specifiers == ["int", "long"] or specifiers == ["long", "int"] or specifiers == ["long"]:
        return LongTypeNode()
    
    raise SyntaxError("Invalid type specifier")

def ast_parse_factor(tokens: List["Token"], min_prec) -> "ExpressionNode":
    token = peek(tokens)
    if token is None:
        raise SyntaxError("Unexpected end of file")

    if isinstance(token, (Constant, LongIntegerConstant)):
        return ast_parse_constant(tokens)
    elif isinstance(token, Identifier) and isinstance(peek(tokens, 2), OpenParenthesis):
        return ast_parse_function_call(tokens)
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
        next_token = peek(tokens, 2)
        if isinstance(next_token, Identifier) and next_token.name in ["long", "int"]:
            expect_and_take(OpenParenthesis, tokens)
            cast_type = ast_get_type(tokens)
            expect_and_take(CloseParenthesis, tokens)
            exp = ast_parse_exp(tokens, min_prec)
            return CastExpressionNode(cast_type, exp)
        else:
            expect_and_take(OpenParenthesis, tokens)
            exp = ast_parse_exp(tokens, 0)
            expect_and_take(CloseParenthesis, tokens)
            return exp
    else:
        raise SyntaxError("Unexpected symbol")


def ast_parse_block_item(tokens: List["Token"]) -> "BlockItemNode":
    match peek(tokens):
        # Declaration
        case Identifier("int" | "long" | "static" | "extern", True): 
            return DeclarationBlockItemNode(ast_parse_declaration(tokens))
        # Statement
        case _:
            return StatementBlockItemNode(ast_parse_statement(tokens))

def ast_parse_declaration(tokens: List["Token"]) -> "DeclarationNode":
    types = []
    storage_specifiers = []
    while isinstance(peek(tokens), Identifier) and peek(tokens).is_keyword:
        if peek(tokens).name in ["int", "long"]:
            types.append(expect_and_take(Identifier, tokens))
        else:
            storage_specifiers.append(expect_and_take(Identifier, tokens))

    if len(types) < 1 or len(storage_specifiers) > 1:
        raise SyntaxError("Wrong declaration!")

    if isinstance(peek(tokens, 2), OpenParenthesis):
        return ast_parse_function_declaration(types, storage_specifiers, tokens)
    else:
        return ast_parse_variable_declaration(types, storage_specifiers, tokens)

def storage_class_from_identifier(identifier: Identifier) -> StorageClassSpecifierNode:
    if identifier.name == "static":
        return StaticStorageClassNode()
    elif identifier.name == "extern":
        return ExternStorageClassNode()
    else:
        raise SyntaxError 
    
def ast_parse_variable_declaration(types: List["Identifier"], storage_specifiers: List["Identifier"], tokens: List["Token"]) -> "VariableDeclarationNode":
    var_type = ast_type_from_tokens(types)
    
    storage_class: Optional["StorageClassSpecifierNode"] = None
    if len(storage_specifiers) == 1:
        storage_class = storage_class_from_identifier(storage_specifiers[0])
     
    decl_identifier: Identifier = expect_and_take(Identifier, tokens)
    if decl_identifier.is_keyword:
        raise SyntaxError
    t = peek(tokens)
    if isinstance(t, Semicolon):
        decl = VariableDeclarationNode(decl_identifier.name, None, var_type, storage_class)
        expect_and_take(Semicolon, tokens)
        return decl
    expect_and_take(EqualSign, tokens)
    exp = ast_parse_exp(tokens, 0)
    decl = VariableDeclarationNode(decl_identifier.name, exp, var_type, storage_class)
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


def ast_parse_for_init(tokens: List["Token"]) -> Optional["ForInitNode"]:
    match peek(tokens):
        case Semicolon():
            expect_and_take(Semicolon, tokens)
            return None
        case Identifier("int" | "long" | "static" | "extern", True):
            decl = ast_parse_declaration(tokens)
            if not isinstance(decl, VariableDeclarationNode):
                raise SyntaxError
            return ForInitDeclarationNode(decl)
        case _:
            exp = ast_parse_exp(tokens, 0)
            expect_and_take(Semicolon, tokens)
            return ForInitExpressionNode(exp)


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


def ast_parse_statement(tokens: List["Token"]) -> "StatementNode":
    match peek(tokens):
        case Semicolon():
            take_token(tokens)
            return NullStatementNode()
        case Identifier("return", True):
            r_statement = ast_parse_return(tokens)
            return r_statement
        case Identifier("goto", True):
            expect_and_take(Identifier, tokens)
            t: Identifier = expect_and_take(Identifier, tokens)
            expect_and_take(Semicolon, tokens)
            return GotoStatement(t.name)
        case Identifier("if", True):
            if_statement = ast_parse_if(tokens)
            return if_statement
        case Identifier("switch", True):
            expect_and_take(Identifier, tokens)
            expect_and_take(OpenParenthesis, tokens)
            exp = ast_parse_exp(tokens, 0)
            expect_and_take(CloseParenthesis, tokens)
            st = ast_parse_statement(tokens)
            return SwitchStatementNode(exp, st, [], None, "")
        case Identifier("case"):
            expect_and_take(Identifier, tokens)
            val = ast_parse_exp(tokens, 0)
            expect_and_take(Colon, tokens)
            st = ast_parse_statement(tokens)
            return CaseLabeledStatement(val, st, "", "")
        case Identifier("default"):
            expect_and_take(Identifier, tokens)
            expect_and_take(Colon, tokens)
            st = ast_parse_statement(tokens)
            return DefaultLabeledStatement(st, "", "")
        case Identifier(name, False) if isinstance(peek(tokens, 2), Colon):
            expect_and_take(Identifier, tokens)
            expect_and_take(Colon, tokens)
            st = ast_parse_statement(tokens)
            return LabeledStatement(name, st)
        case Identifier("break", True):
            expect_and_take(Identifier, tokens)
            expect_and_take(Semicolon, tokens)
            return BreakStatementNode("")
        case Identifier("continue", True):
            expect_and_take(Identifier, tokens)
            expect_and_take(Semicolon, tokens)
            return ContinueStatementNode("")
        case Identifier("while", True):
            while_loop = ast_parse_while_loop(tokens)
            return while_loop
        case Identifier("do", True):
            do_while_loop = ast_parse_do_while_loop(tokens)
            return do_while_loop
        case Identifier("for", True):
            for_loop = ast_parse_for_loop(tokens)
            return for_loop
        case OpenBrace():
            block = ast_parse_block(tokens)
            return CompoundStatement(block)
        case _:
            exp = ast_parse_exp(tokens, 0)
            expect_and_take(Semicolon, tokens)
            return ExpressionStatementNode(exp)

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


def ast_parse_function_declaration(types: List["Identifier"], storage_specifiers: List["Identifier"], tokens: List["Token"]) -> "FunctionDeclarationNode":
    fun_ret_type = ast_type_from_tokens(types)
    
    storage_class: Optional["StorageClassSpecifierNode"] = None
    if len(storage_specifiers) == 1:
        storage_class = storage_class_from_identifier(storage_specifiers[0])

    expect(Identifier, tokens)
    f_name: "Identifier" = take_token(tokens)

    if f_name.is_keyword:
        raise SyntaxError

    expect_and_take(OpenParenthesis, tokens)

    param_names = None
    param_types = None    

    if check_identifier("void", True, peek(tokens)):
        expect_and_take(Identifier, tokens)
        param_names = []
        param_types = []
    else:
        param_type = ast_get_type(tokens)
        p: Identifier = expect_and_take(Identifier, tokens)
        param_names = [p.name]
        param_types = [param_type]
        while not isinstance(peek(tokens), CloseParenthesis):
            expect_and_take(Comma, tokens)
            param_type = ast_get_type(tokens)
            p: Identifier = expect_and_take(Identifier, tokens)
            param_names.append(p.name)
            param_types.append(param_type)
            
    expect_and_take(CloseParenthesis, tokens)

    f_body = None

    if not isinstance(peek(tokens), Semicolon):
        f_body = ast_parse_block(tokens)
    else:
        expect_and_take(Semicolon, tokens)

    fun_type = FunTypeNode(param_types, fun_ret_type)
    return FunctionDeclarationNode(f_name.name, param_names, f_body, fun_type, storage_class)


def ast_parse_program(tokens: List["Token"]) -> "ProgramNode":
    declarations = []

    while tokens:
        decl = ast_parse_declaration(tokens)
        declarations.append(decl)

    p_node = ProgramNode(declarations)
    return p_node

def parse(tokens: List["Token"]) -> "ProgramNode":
    return ast_parse_program(tokens)