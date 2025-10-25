from tacky import *
from parser import *


block_counter = 0


def get_block_id():
    global block_counter
    block_counter += 1
    return block_counter


def resolve_variables(node: AstNode, block_id: int, variable_map: Dict[str, str]) -> AstNode:
    if node is None:
        return None

    match node:
        # Top level
        case ProgramNode(f_node):
            f_node_resolved = resolve_variables(f_node, block_id, variable_map)
            p_node = ProgramNode(f_node_resolved)
            return p_node
        case FunctionNode(name, BlockNode(block_items)):
            block_items_resolved = []
            for bi in block_items:
                block_items_resolved.append(resolve_variables(bi, block_id, variable_map))
            f_node = FunctionNode(name, BlockNode(block_items_resolved))
            return f_node
        case DeclarationNode(name, exp):
            if name in variable_map and variable_map[name][1] == block_id:
                raise SyntaxError("Duplicate variable declaration!")
            name_resolved = get_temp_var_name(name)
            variable_map[name] = (name_resolved, block_id)
            d_node = DeclarationNode(name_resolved, resolve_variables(exp, block_id, variable_map))
            return d_node
        case BlockNode(items):
            new_block_id = get_block_id()
            variable_map_copy = variable_map.copy()
            items = [resolve_variables(item, new_block_id, variable_map_copy) for item in items]
            return BlockNode(items)
        
        # Block item nodes
        case StatementBlockItemNode(statement):
            return StatementBlockItemNode(resolve_variables(statement, block_id, variable_map))
        case DeclarationBlockItemNode(declaration):
            return DeclarationBlockItemNode(resolve_variables(declaration, block_id, variable_map))
        
        # Statements
        case NullStatementNode():
            return node
        case ReturnStatementNode(exp):
            return ReturnStatementNode(resolve_variables(exp, block_id, variable_map))
        case ExpressionStatementNode(exp):
            return ExpressionStatementNode(resolve_variables(exp, block_id, variable_map))
        case IfStatementNode(cond, then_exp, else_exp):
            return IfStatementNode(
                resolve_variables(cond, block_id, variable_map),
                resolve_variables(then_exp, block_id, variable_map),
                resolve_variables(else_exp, block_id, variable_map)
            )
        case GotoStatement(_):
            return node
        case LabeledStatement(name, statement):
            return LabeledStatement(name, resolve_variables(statement, block_id, variable_map))
        case CompoundStatement(block):
            return CompoundStatement(resolve_variables(block, block_id, variable_map))
        case BreakStatementNode(_):
            return node
        case ContinueStatementNode(_):
            return node
        
        case WhileStatementNode(cond, body, label):
            n_cond = resolve_variables(cond, block_id, variable_map)
            n_body = resolve_variables(body, block_id, variable_map)
            return WhileStatementNode(n_cond, n_body, label)
        case DoWhileStatementNode(body, cond, label):
            n_cond = resolve_variables(cond, block_id, variable_map)
            n_body = resolve_variables(body, block_id, variable_map)
            return DoWhileStatementNode(n_body, n_cond, label)
        case ForStatementNode(init, cond, post, body, label):
            new_block_id = get_block_id()
            new_variable_map = variable_map.copy()
            n_init = resolve_variables(init, new_block_id, new_variable_map)
            n_cond = resolve_variables(cond, new_block_id, new_variable_map) if cond else None
            n_post = resolve_variables(post, new_block_id, new_variable_map) if post else None
            n_body = resolve_variables(body, new_block_id, new_variable_map)
            return ForStatementNode(n_init, n_cond, n_post, n_body, label)
        case ForInitDeclarationNode(decl):
            n_decl = resolve_variables(decl, block_id, variable_map)
            return ForInitDeclarationNode(n_decl)
        case ForInitExpressionNode(exp):
            n_exp = resolve_variables(exp, block_id, variable_map)
            return ForInitExpressionNode(n_exp)
        case SwitchStatementNode(exp, body, cases, label):
            n_exp = resolve_variables(exp, block_id, variable_map)
            n_body = resolve_variables(body, block_id, variable_map)
            return SwitchStatementNode(n_exp, n_body, cases, label)
        case CaseLabeledStatement(val, st, label):
            # val has to be a constant expression so we don't resolve variables for it
            n_st = resolve_variables(st, block_id, variable_map)
            return CaseLabeledStatement(val, n_st, label)
        case DefaultLabeledStatement(st, label):
            n_st = resolve_variables(st, block_id, variable_map)
            return DefaultLabeledStatement(n_st, label)

        # Expressions
        case ConstantExpressionNode(_):
            return node
        case UnaryExpressionNode(unop, exp):
            return UnaryExpressionNode(unop, resolve_variables(exp, block_id, variable_map))
        case BinaryExpressionNode(binop, exp1, exp2):
            return BinaryExpressionNode(
                binop,
                resolve_variables(exp1, block_id, variable_map),
                resolve_variables(exp2, block_id, variable_map)
            )
        case PrefixExpressionNode(op, exp):
            return PrefixExpressionNode(op, resolve_variables(exp, block_id, variable_map))
        case PostfixExpressionNode(op, exp):
            return PostfixExpressionNode(op, resolve_variables(exp, block_id, variable_map))
        case VariableExpressionNode(name):
            if not name in variable_map:
                raise SyntaxError("Undeclared variable!")
            return VariableExpressionNode(variable_map[name][0])
        case AssignmentExpressionNode(lhs, rhs):
            if not isinstance(lhs, VariableExpressionNode):
                raise SyntaxError("Invalid lvalue!")
            return AssignmentExpressionNode(
                resolve_variables(lhs, block_id, variable_map),
                resolve_variables(rhs, block_id, variable_map)
            )
        case CompoundAssignmentExpressionNode(binop, lhs, rhs):
            if not isinstance(lhs, VariableExpressionNode):
                raise SyntaxError("Invalid lvalue!")
            return CompoundAssignmentExpressionNode(
                binop,
                resolve_variables(lhs, block_id, variable_map),
                resolve_variables(rhs, block_id, variable_map)
            )
        case ConditionalExpressionNode(cond, true_exp, false_exp):
            return ConditionalExpressionNode(
                resolve_variables(cond, block_id, variable_map),
                resolve_variables(true_exp, block_id, variable_map),
                resolve_variables(false_exp, block_id, variable_map)
            )
        case _:
            raise SyntaxError("Unknown AST node!")

def resolve_labels(node: AstNode, update_goto: bool, func_prefix: str, label_map: Dict[str, str]) -> AstNode:
    if node is None:
        return None

    match node:
        # Top level
        case ProgramNode(f_node):
            f_node_resolved = resolve_labels(f_node, update_goto, func_prefix, label_map)
            p_node = ProgramNode(f_node_resolved)
            return p_node
        case FunctionNode(name, block):
            return FunctionNode(name, resolve_labels(block, update_goto, func_prefix, label_map))
        case BlockNode(block_items):
            block_items_resolved = []
            for bi in block_items:
                block_items_resolved.append(resolve_labels(bi, update_goto, func_prefix, label_map))
            return BlockNode(block_items_resolved)
        
        # Block item nodes
        case StatementBlockItemNode(statement):
            return StatementBlockItemNode(resolve_labels(statement, update_goto, func_prefix, label_map))
        
        case IfStatementNode(cond, then_exp, else_exp):
            return IfStatementNode(
                cond,
                resolve_labels(then_exp, update_goto, func_prefix, label_map),
                resolve_labels(else_exp, update_goto, func_prefix, label_map)
            )
        case WhileStatementNode(cond, body, label):
            return WhileStatementNode(cond, resolve_labels(body, update_goto, func_prefix, label_map), label)
        case DoWhileStatementNode(body, cond, label):
            return DoWhileStatementNode(resolve_labels(body, update_goto, func_prefix, label_map), cond, label)
        case ForStatementNode(init, cond, post, body, label):
            return ForStatementNode(
                init,
                cond,
                post,
                resolve_labels(body, update_goto, func_prefix, label_map),
                label
            )
        case GotoStatement(label):
            if not update_goto:
                return node
            fname = f"{func_prefix}.{label}"
            if fname not in label_map:
                raise SyntaxError("Unknown label!")
            return GotoStatement(label_map[fname])
        case LabeledStatement(name, statement):
            if update_goto:
                return LabeledStatement(name, resolve_labels(statement, update_goto, func_prefix, label_map))
            fname = f"{func_prefix}.{name}"
            if fname in label_map:
                raise SyntaxError("Duplicate labels!")
            new_name = get_label_name(fname)
            label_map[fname] = new_name
            return LabeledStatement(new_name, resolve_labels(statement, update_goto, func_prefix, label_map))
        case SwitchStatementNode(exp, st, cases, label):
            return SwitchStatementNode(exp, resolve_labels(st, update_goto, func_prefix, label_map), cases, label)
        case CaseLabeledStatement(val, st, label):
            return CaseLabeledStatement(val, resolve_labels(st, update_goto, func_prefix, label_map), label)
        case DefaultLabeledStatement(st, label):
            return DefaultLabeledStatement(resolve_labels(st, update_goto, func_prefix, label_map), label)
        case CompoundStatement(block):
            return CompoundStatement(resolve_labels(block, update_goto, func_prefix, label_map))
        case _:
            return node

# def traverse_ast(node: AstNode, params: dict, func: Callable[[AstNode, dict], Optional[AstNode]]) -> AstNode:
#     processed = func(node, params)
#     result = processed if processed is not None else node
#     fs = vars(result)

#     for f_name in fs.keys():
#         f_val = fs[f_name]
#         if isinstance(f_val, AstNode):
#             fs[f_name] = traverse_ast(f_val, params, func)
#         elif isinstance(f_val, list):
#             for el in f_val:
#                 if isinstance(el, AstNode):
#                     traverse_ast(el, params, func)

#     return result

def label_break_statements(node: AstNode, labels: List[Tuple["str", "str"]]) -> AstNode:   
    match node:
        # Top Level
        case ProgramNode(func):
            n_func = label_break_statements(func, labels)
            return ProgramNode(n_func)
        case FunctionNode(name, body):
            n_body = label_break_statements(body, labels)
            return FunctionNode(name, n_body)
        case DeclarationNode(_, _):
            return node
        
        # Block Items
        case StatementBlockItemNode(statement):
            n_statement = label_break_statements(statement, labels)
            return StatementBlockItemNode(n_statement)
        case DeclarationBlockItemNode(declaration):
            n_declaration = label_break_statements(declaration, labels)
            return DeclarationBlockItemNode(n_declaration)
        case BlockNode(items):
            n_items = []
            for item in items:
                n_items.append(label_break_statements(item, labels))
            return BlockNode(n_items)
        
        # Statements
        case ReturnStatementNode(_) | ExpressionStatementNode(_) | GotoStatement(_) | NullStatementNode():
            return node
        case LabeledStatement(name, statement):
            return LabeledStatement(name, label_break_statements(statement, labels))
        case CaseLabeledStatement(val, st, _):
            labels_copy = labels[:]
            while labels_copy and labels_copy[-1][1] != "switch":
                labels_copy.pop()
            if not labels_copy:
                raise SyntaxError
            return CaseLabeledStatement(val, label_break_statements(st, labels), labels_copy[-1][0])
        case DefaultLabeledStatement(st, label):
            labels_copy = labels[:]
            while labels_copy and labels_copy[-1][1] != "switch":
                labels_copy.pop()
            if not labels_copy:
                raise SyntaxError
            return DefaultLabeledStatement(label_break_statements(st, labels), labels_copy[-1][0])
        case IfStatementNode(cond, then_st, else_st):
            n_then_st = label_break_statements(then_st, labels)
            n_else_st = label_break_statements(else_st, labels) if else_st else None
            return IfStatementNode(cond, n_then_st, n_else_st)
        case CompoundStatement(block):
            n_block = label_break_statements(block, labels)
            return CompoundStatement(n_block)
        case WhileStatementNode(cond, body, _):
            loop_label = get_label_name("while")
            n_body = label_break_statements(body, labels + [(loop_label, "loop")])
            return WhileStatementNode(cond, n_body, loop_label)
        case DoWhileStatementNode(body, cond, label):
            loop_label = get_label_name("dowhile")
            n_body = label_break_statements(body, labels + [(loop_label, "loop")])
            return DoWhileStatementNode(n_body, cond, loop_label)
        case ForStatementNode(init, cond, post, body, _):
            loop_label = get_label_name("for")
            n_body = label_break_statements(body, labels + [(loop_label, "loop")])
            return ForStatementNode(init, cond, post, n_body, loop_label)
        case SwitchStatementNode(exp, body, cases, _):
            switch_label = get_label_name("switch")
            n_body = label_break_statements(body, labels + [(switch_label, "switch")])
            return SwitchStatementNode(exp, n_body, cases, switch_label)
        case BreakStatementNode(_):
            if not labels:
                raise SyntaxError
            return BreakStatementNode(labels[-1][0])
        case ContinueStatementNode(_):
            labels_copy = labels[:]
            while labels_copy and labels_copy[-1][1] != "loop":
                labels_copy.pop()
            if not labels_copy:
                raise SyntaxError
            return ContinueStatementNode(labels_copy[-1])
        case _:
            raise SyntaxError


def validate(ast: AstNode) -> "AstNode":
    s1 = resolve_variables(ast, 0, dict())
    label_map = dict()
    s2 = resolve_labels(s1, False, "", label_map)
    s3 = resolve_labels(s2, True, "", label_map)
    s4 = label_break_statements(s3, [])
    # traverse_ast(s3, {}, lambda node: print(node))
    return s4