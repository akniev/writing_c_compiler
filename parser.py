from lexer import *
from dataclasses import *
from parser_types import *



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
    return UnaryExpressionNode(None, get_unary_operator(token), exp)

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
            left = AssignmentExpressionNode(None, left, right)
        elif isinstance(next_token, QuestionMark):
            middle = ast_parse_conditional_middle(tokens)
            right = ast_parse_exp(tokens, precedence(next_token))
            left = ConditionalExpressionNode(None, left, middle, right)
        elif type(next_token) in [TwoPlusses, TwoMinuses]:
            t = take_token(tokens)
            operator = ast_parse_prefix_postfix_unop(t)
            left = PostfixExpressionNode(operator, left)
        elif type(next_token) in COMPOUND_ASSIGNMENT_TOKENS:
            t = take_token(tokens)
            operator = ast_parse_compop(t)
            right = ast_parse_exp(tokens, precedence(next_token))
            left = CompoundAssignmentExpressionNode(None, operator, left, right)
        else:
            operator = ast_parse_binop(tokens)
            right = ast_parse_exp(tokens, precedence(next_token) + 1)
            left = BinaryExpressionNode(None, operator, left, right)
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

    return FunctionCallExpressionNode(None, f_name, f_args, False)

def ast_parse_constant(tokens: List[Token]) -> ConstIntExpressionNode | ConstLongExpressionNode:
    token = take_token(tokens)
    if isinstance(token, IntConstant):
        if token.value > 2**63 - 1:
            raise SyntaxError("Constant is too large to represent as an int or long")
        elif token.value <= 2**31 - 1:
            return ConstIntExpressionNode(None, token.value)
        else:
            return ConstLongExpressionNode(None, token.value)
    elif isinstance(token, LongConstant):
        if token.value > 2**63 - 1:
            raise SyntaxError("Constant is too large to represent as an int or long")
        else:
            return ConstLongExpressionNode(None, token.value)
    else:
        raise SyntaxError("Wrong constant token!")

def ast_parse_factor(tokens: List["Token"], min_prec) -> "ExpressionNode":
    token = peek(tokens)
    if token is None:
        raise SyntaxError("Unexpected end of file")

    if isinstance(token, IntConstant) or isinstance(token, LongConstant):
        const_token = ast_parse_constant(tokens)
        return const_token
    elif isinstance(token, OpenParenthesis) and has_type_specifier(tokens[1:]):
        type_specifiers = []
        expect_and_take(OpenParenthesis, tokens)
        while has_type_specifier(tokens):
            type_specifier = ast_parse_specifier(tokens)
            type_specifiers.append(type_specifier)
        expect_and_take(CloseParenthesis, tokens)
        exp = ast_parse_exp(tokens, min_prec)
        target_type = parse_type_specifiers(type_specifiers)
        return CastExpressionNode(None, target_type, exp)
    elif isinstance(token, Identifier) and isinstance(peek(tokens, 2), OpenParenthesis):
        return ast_parse_function_call(tokens)
    elif isinstance(token, Identifier):
        t: Identifier = token
        if t.is_keyword:
            raise SyntaxError
        take_token(tokens)
        return VariableExpressionNode(None, t.name)
    elif is_unary(token):
        return ast_parse_unary_expression(tokens, min_prec)
    elif is_prefix_postfix(token):
        t = take_token(tokens)
        op = ast_parse_prefix_postfix_unop(t)
        exp = ast_parse_exp(tokens, 1000)
        return PrefixExpressionNode(None, op, exp)
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
        case Identifier("int" | "long" | "static" | "extern", True): 
            return DeclarationBlockItemNode(ast_parse_declaration(tokens))
        # Statement
        case _:
            return StatementBlockItemNode(ast_parse_statement(tokens))

def has_specifier(tokens) -> bool:
    next_token = peek(tokens)
    if isinstance(next_token, Identifier) and next_token.is_keyword and next_token.name  in ["static", "extern", "int", "long"]:
        return True
    return False

def has_type_specifier(tokens) -> bool:
    next_token = peek(tokens)
    if isinstance(next_token, Identifier) and next_token.is_keyword and next_token.name in ["int", "long"]:
        return True
    return False

def ast_parse_specifier(tokens) -> Token:
    next_token = take_token(tokens)
    if not (isinstance(next_token, Identifier) and next_token.is_keyword):
        raise SyntaxError("Wrong specifier!")
    return next_token
    
def parse_type_and_storage_class(specifiers: List[Identifier]) -> Tuple[List[TypeNode], List[DeclarationStorageClass]]:
    types = []
    storage_classes = []
    for specifier in specifiers:
        if specifier.name in ["int", "long"]:
            types.append(specifier)
        elif specifier.name in ["static", "extern"]:
            storage_classes.append(specifier)
        else:
            raise SyntaxError("Wrong specifier!")
    return (types, storage_classes)


def ast_parse_declaration(tokens: List["Token"]) -> "DeclarationNode":
    specifiers: List[Identifier] = []
    while has_specifier(tokens):
        specifier = ast_parse_specifier(tokens)
        specifiers.append(specifier)
    
    identifier: Identifier = expect_and_take(Identifier, tokens)
    if identifier.is_keyword:
        SyntaxError("Function/Variable names cannot be keywords!")
    
    if isinstance(peek(tokens), OpenParenthesis):
        return ast_parse_function_declaration(specifiers, identifier, tokens)
    else:
        return ast_parse_variable_declaration(specifiers, identifier, tokens)


def storage_class_from_identifier(identifier: Identifier) -> DeclarationStorageClass:
    if identifier.name == "static":
        return StaticStorageClass()
    elif identifier.name == "extern":
        return ExternStorageClass()
    else:
        raise SyntaxError 
    
def ast_parse_variable_declaration(specifiers: List["Identifier"], identifier: Identifier, tokens: List["Token"]) -> "VariableDeclarationNode":
    type_specifiers, storage_specifiers = parse_type_and_storage_class(specifiers)
    
    var_type = parse_type_specifiers(type_specifiers)

    storage_class: Optional["DeclarationStorageClass"] = None
    if len(storage_specifiers) == 1:
        storage_class = storage_class_from_identifier(storage_specifiers[0])
    elif len(storage_specifiers) > 1:
        raise SyntaxError("Variable declaration must have no more than 1 storage specifiers!")
    
    if identifier.is_keyword:
        raise SyntaxError("Variable name must not be a keyword!")
    t = peek(tokens)
    if isinstance(t, Semicolon):
        decl = VariableDeclarationNode(identifier.name, None, var_type, storage_class)
        expect_and_take(Semicolon, tokens)
        return decl
    expect_and_take(EqualSign, tokens)
    exp = ast_parse_exp(tokens, 0)
    decl = VariableDeclarationNode(identifier.name, exp, var_type, storage_class)
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

def parse_type_specifiers(specifiers: List[Token]):
    match specifiers:
        case [Identifier("int", True)]:
            return IntTypeNode()
        case [Identifier("long", True)] | [Identifier("long", True), Identifier("int", True)] | [Identifier("int", True), Identifier("long", True)]:
            return LongTypeNode()
        case _:
            raise SyntaxError("Wrong type specifiers!")

def type_from_token(token: Token) -> TypeNode:
    match token:
        case Identifier(_, "int", True):
            return IntTypeNode()
        case Identifier(_, "long", True):
            return LongTypeNode()
        case _:
            raise SyntaxError("Wrong token type!")

def ast_parse_type_specifiers(tokens: List[Token]) -> TypeNode:
    type_specifiers = []
    while has_type_specifier(tokens):
        type_specifiers.append(take_token(tokens))
    return parse_type_specifiers(type_specifiers)


def ast_parse_function_declaration(specifiers: List["Identifier"], identifier: Identifier, tokens: List["Token"]) -> "FunctionDeclarationNode":
    type_specifiers, storage_specifiers = parse_type_and_storage_class(specifiers)
    
    ret_type = parse_type_specifiers(type_specifiers)
    
    storage_class: Optional["DeclarationStorageClass"] = None
    if len(storage_specifiers) == 1:
        storage_class = storage_class_from_identifier(storage_specifiers[0])
    elif len(storage_specifiers) > 1:
        raise SyntaxError("Function declaration must have no more than 1 storage specifiers!")
    
    if identifier.is_keyword:
        raise SyntaxError("Function name must not be a keyword!")

    expect_and_take(OpenParenthesis, tokens)

    param_list: List[TypeNode] = []
    param_types = None

    if check_identifier("void", True, peek(tokens)):
        expect_and_take(Identifier, tokens)
        param_list = []
        param_types = []
    else:
        type_token = ast_parse_type_specifiers(tokens)
        p: Identifier = expect_and_take(Identifier, tokens)
        param_list = [p.name]
        param_types = [type_token]
        while not isinstance(peek(tokens), CloseParenthesis):
            expect_and_take(Comma, tokens)
            type_token = ast_parse_type_specifiers(tokens)
            p: Identifier = expect_and_take(Identifier, tokens)
            param_list.append(p.name)
            param_types.append(type_token)
 
    expect_and_take(CloseParenthesis, tokens)

    f_body = None

    if not isinstance(peek(tokens), Semicolon):
        f_body = ast_parse_block(tokens)
    else:
        expect_and_take(Semicolon, tokens)

    fun_type = FunTypeNode(param_types, ret_type)

    return FunctionDeclarationNode(identifier.name, param_list, f_body, fun_type, storage_class)


def ast_parse_program(tokens: List["Token"]) -> "ProgramNode":
    declarations = []

    while tokens:
        decl = ast_parse_declaration(tokens)
        declarations.append(decl)

    p_node = ProgramNode(declarations)
    return p_node

def parse(tokens: List["Token"]) -> "ProgramNode":
    return ast_parse_program(tokens)