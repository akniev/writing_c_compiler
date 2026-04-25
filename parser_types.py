from lexer import *
from dataclasses import *

# MARK: Tokens

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

# MARK: Precedence

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


# MARK: Top level        

@dataclass
class ProgramNode(AstNode):
    declarations: List["DeclarationNode"]



# MARK: Declaration

class DeclarationStorageClass(AstNode):
    pass

class StaticStorageClass(DeclarationStorageClass):
    pass

class ExternStorageClass(DeclarationStorageClass):
    pass


class DeclarationNode(AstNode):
    pass

@dataclass
class VariableDeclarationNode(DeclarationNode):
    name: str
    init: Optional["ExpressionNode"]
    storage_class: Optional["DeclarationStorageClass"]

@dataclass
class FunctionDeclarationNode(AstNode):
    name: str
    params: List[str]
    body: Optional["BlockNode"]
    storage_class: Optional["DeclarationStorageClass"]


# MARK: Block items

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




# MARK: Statements

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



# MARK: ForInit

class ForInitNode(AstNode):
    pass

@dataclass
class ForInitDeclarationNode(ForInitNode):
    declaration: "VariableDeclarationNode"

@dataclass
class ForInitExpressionNode(ForInitNode):
    expression: Optional["ExpressionNode"]


# MARK: Unary Operators

class UnaryOperatorNode(AstNode):
    pass

class ComplementOperatorNode(UnaryOperatorNode):
    pass

class NegateOperatorNode(UnaryOperatorNode):
    pass

class NotOperatorNode(UnaryOperatorNode):
    pass


# MARK: Prefix/Postfix Operators

class IncrementOperatorNode(UnaryOperatorNode):
    pass

class DecrementOperatorNode(UnaryOperatorNode):
    pass




# MARK: Binary Operators

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


# MARK: Expressions

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

@dataclass
class FunctionCallExpressionNode(ExpressionNode):
    name: str
    args: List["ExpressionNode"]
    plt: bool