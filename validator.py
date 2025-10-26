from tacky import *
from parser import *


block_counter = 0


def get_block_id():
    global block_counter
    block_counter += 1
    return block_counter

@dataclass
class IdentifierMapEntry:
    new_name: str
    block_id: bool
    has_linkage: bool

def resolve_param(param: str, block_id: int, identifier_map: Dict[str, IdentifierMapEntry]) -> str:
    if param in identifier_map and identifier_map[param].block_id == block_id:
        raise SyntaxError("Duplicate variable declaration!")
    param_resolved = get_temp_var_name(param)
    identifier_map[param] = IdentifierMapEntry(param_resolved, block_id, False)
    return param_resolved

def identifier_resolution(node: AstNode, block_id: int, identifier_map: Dict[str, IdentifierMapEntry]) -> AstNode:
    if node is None:
        return None

    match node:
        # Top level
        case ProgramNode(fun_decls):
            new_fun_decls = []
            for fun_decl in fun_decls:
                f_decl_resolved = identifier_resolution(fun_decl, block_id, identifier_map)
            new_fun_decls.append(f_decl_resolved)
            p_node = ProgramNode(new_fun_decls)
            return p_node
        
        # Declarations
        case FunctionDeclarationNode(name, params, body):
            new_block_id = get_block_id()
            if name in identifier_map:
                prev_entry = identifier_map[name]
                if prev_entry.block_id == block_id and (not prev_entry.has_linkage):
                    raise SyntaxError("Duplicate declaration!")
            identifier_map[name] = IdentifierMapEntry(name, block_id, True)
            
            inner_map = identifier_map.copy()
            new_params = []
            for param in params:
                new_params.append(resolve_param(param, new_block_id, inner_map))
            
            new_body: BlockNode|None = None
            if body is not None and block_id > 0:
                raise SyntaxError("Illegal function definition!")
            if body is not None:
                body_items = body.items
                new_body_items = []
                for body_item in body_items:
                    new_body_item = identifier_resolution(body_item, new_block_id, inner_map)
                    new_body_items.append(new_body_item)
                new_body = BlockNode(new_body_items)
            return FunctionDeclarationNode(name, new_params, new_body)
        case VariableDeclarationNode(name, exp):
            new_name = resolve_param(name, block_id, identifier_map)
            # if name in identifier_map and identifier_map[name].block_id == block_id:
            #     raise SyntaxError("Duplicate variable declaration!")
            # name_resolved = get_temp_var_name(name)
            # identifier_map[name] = IdentifierMapEntry(name_resolved, block_id, False)
            d_node = VariableDeclarationNode(new_name, identifier_resolution(exp, block_id, identifier_map))
            return d_node
        
        
        # Blocks
        case BlockNode(items):
            new_block_id = get_block_id()
            variable_map_copy = identifier_map.copy()
            items = [identifier_resolution(item, new_block_id, variable_map_copy) for item in items]
            return BlockNode(items)
        
        case StatementBlockItemNode(statement):
            return StatementBlockItemNode(identifier_resolution(statement, block_id, identifier_map))
        case DeclarationBlockItemNode(declaration):
            return DeclarationBlockItemNode(identifier_resolution(declaration, block_id, identifier_map))
        
        # Statements
        case NullStatementNode():
            return node
        case ReturnStatementNode(exp):
            return ReturnStatementNode(identifier_resolution(exp, block_id, identifier_map))
        case ExpressionStatementNode(exp):
            return ExpressionStatementNode(identifier_resolution(exp, block_id, identifier_map))
        case IfStatementNode(cond, then_exp, else_exp):
            return IfStatementNode(
                identifier_resolution(cond, block_id, identifier_map),
                identifier_resolution(then_exp, block_id, identifier_map),
                identifier_resolution(else_exp, block_id, identifier_map)
            )
        case GotoStatement(_):
            return node
        case LabeledStatement(name, statement):
            return LabeledStatement(name, identifier_resolution(statement, block_id, identifier_map))
        case CompoundStatement(block):
            return CompoundStatement(identifier_resolution(block, block_id, identifier_map))
        case BreakStatementNode(_):
            return node
        case ContinueStatementNode(_):
            return node
        
        case WhileStatementNode(cond, body, label):
            n_cond = identifier_resolution(cond, block_id, identifier_map)
            n_body = identifier_resolution(body, block_id, identifier_map)
            return WhileStatementNode(n_cond, n_body, label)
        case DoWhileStatementNode(body, cond, label):
            n_cond = identifier_resolution(cond, block_id, identifier_map)
            n_body = identifier_resolution(body, block_id, identifier_map)
            return DoWhileStatementNode(n_body, n_cond, label)
        case ForStatementNode(init, cond, post, body, label):
            new_block_id = get_block_id()
            new_variable_map = identifier_map.copy()
            n_init = identifier_resolution(init, new_block_id, new_variable_map)
            n_cond = identifier_resolution(cond, new_block_id, new_variable_map) if cond else None
            n_post = identifier_resolution(post, new_block_id, new_variable_map) if post else None
            n_body = identifier_resolution(body, new_block_id, new_variable_map)
            return ForStatementNode(n_init, n_cond, n_post, n_body, label)
        case ForInitDeclarationNode(decl):
            n_decl = identifier_resolution(decl, block_id, identifier_map)
            return ForInitDeclarationNode(n_decl)
        case ForInitExpressionNode(exp):
            n_exp = identifier_resolution(exp, block_id, identifier_map)
            return ForInitExpressionNode(n_exp)
        case SwitchStatementNode(exp, body, cases, defaultCase, label):
            n_exp = identifier_resolution(exp, block_id, identifier_map)
            n_body = identifier_resolution(body, block_id, identifier_map)
            return SwitchStatementNode(n_exp, n_body, cases, defaultCase, label)
        case CaseLabeledStatement(val, st, switch_label, label):
            # val has to be a constant expression so we don't resolve variables for it
            n_st = identifier_resolution(st, block_id, identifier_map)
            return CaseLabeledStatement(val, n_st, switch_label, label)
        case DefaultLabeledStatement(st, switch_label, label):
            n_st = identifier_resolution(st, block_id, identifier_map)
            return DefaultLabeledStatement(n_st, switch_label, label)

        # Expressions
        case ConstantExpressionNode(_):
            return node
        case UnaryExpressionNode(unop, exp):
            return UnaryExpressionNode(unop, identifier_resolution(exp, block_id, identifier_map))
        case BinaryExpressionNode(binop, exp1, exp2):
            return BinaryExpressionNode(
                binop,
                identifier_resolution(exp1, block_id, identifier_map),
                identifier_resolution(exp2, block_id, identifier_map)
            )
        case PrefixExpressionNode(op, exp):
            return PrefixExpressionNode(op, identifier_resolution(exp, block_id, identifier_map))
        case PostfixExpressionNode(op, exp):
            return PostfixExpressionNode(op, identifier_resolution(exp, block_id, identifier_map))
        case VariableExpressionNode(name):
            if not name in identifier_map:
                raise SyntaxError("Undeclared variable!")
            return VariableExpressionNode(identifier_map[name].new_name)
        case AssignmentExpressionNode(lhs, rhs):
            if not isinstance(lhs, VariableExpressionNode):
                raise SyntaxError("Invalid lvalue!")
            return AssignmentExpressionNode(
                identifier_resolution(lhs, block_id, identifier_map),
                identifier_resolution(rhs, block_id, identifier_map)
            )
        case CompoundAssignmentExpressionNode(binop, lhs, rhs):
            if not isinstance(lhs, VariableExpressionNode):
                raise SyntaxError("Invalid lvalue!")
            return CompoundAssignmentExpressionNode(
                binop,
                identifier_resolution(lhs, block_id, identifier_map),
                identifier_resolution(rhs, block_id, identifier_map)
            )
        case ConditionalExpressionNode(cond, true_exp, false_exp):
            return ConditionalExpressionNode(
                identifier_resolution(cond, block_id, identifier_map),
                identifier_resolution(true_exp, block_id, identifier_map),
                identifier_resolution(false_exp, block_id, identifier_map)
            )
        case FunctionCallExpressionNode(name, args):
            if name in identifier_map:
                new_name = identifier_map[name].new_name
                new_args = []
                for arg in args:
                    new_args.append(identifier_resolution(arg, block_id, identifier_map))
                return FunctionCallExpressionNode(new_name, new_args)
            else:
                raise SyntaxError("Undeclared function!")
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
        case FunctionDeclarationNode(name, block):
            return FunctionDeclarationNode(name, resolve_labels(block, update_goto, func_prefix, label_map))
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
        case SwitchStatementNode(exp, st, cases, defaultCase, label):
            return SwitchStatementNode(exp, resolve_labels(st, update_goto, func_prefix, label_map), cases, defaultCase, label)
        case CaseLabeledStatement(val, st, switch_label, label):
            return CaseLabeledStatement(val, resolve_labels(st, update_goto, func_prefix, label_map), switch_label, label)
        case DefaultLabeledStatement(st, switch_label, label):
            return DefaultLabeledStatement(resolve_labels(st, update_goto, func_prefix, label_map), switch_label, label)
        case CompoundStatement(block):
            return CompoundStatement(resolve_labels(block, update_goto, func_prefix, label_map))
        case _:
            return node

def process_ast(node: AstNode, params: dict, func: Callable[[AstNode, dict], Optional[AstNode]]) -> AstNode:
    processed = func(node, params)
    result = processed if processed is not None else node
    fs = vars(result)

    for f_name, f_val in list(fs.items()):
        if isinstance(f_val, AstNode):
            fs[f_name] = process_ast(f_val, params, func)
        elif isinstance(f_val, list):
            for el in f_val:
                if isinstance(el, AstNode):
                    process_ast(el, params, func)

    return result

def traverse_ast(node: AstNode, params: dict, before: Callable[[AstNode, dict], None], after: Callable[[AstNode, dict], None]):
    if before:
        before(node, params)
    fs = vars(node)
    for _, ch_node in list(fs.items()):
        if isinstance(ch_node, AstNode):
            traverse_ast(ch_node, params, before, after)
        elif isinstance(ch_node, (list, tuple, set)):
            for el in ch_node:
                if isinstance(el, AstNode):
                    traverse_ast(el, params, before, after)
    if after:
        after(node, params)


def label_break_and_continue_statements(node: AstNode, labels: List[Tuple["str", "str"]]) -> AstNode:   
    match node:
        # Top Level
        case ProgramNode(fun_decls):
            new_fun_decls = []
            for fun_decl in fun_decls:
                new_fun_decl = label_break_and_continue_statements(fun_decl, labels)
                new_fun_decls.append(new_fun_decl)
            return ProgramNode(new_fun_decls)
        case FunctionDeclarationNode(name, params, body):
            n_body = label_break_and_continue_statements(body, labels)
            return FunctionDeclarationNode(name, params, n_body)
        case VariableDeclarationNode(_, _):
            return node
        
        # Block Items
        case StatementBlockItemNode(statement):
            n_statement = label_break_and_continue_statements(statement, labels)
            return StatementBlockItemNode(n_statement)
        case DeclarationBlockItemNode(declaration):
            n_declaration = label_break_and_continue_statements(declaration, labels)
            return DeclarationBlockItemNode(n_declaration)
        case BlockNode(items):
            n_items = []
            for item in items:
                n_items.append(label_break_and_continue_statements(item, labels))
            return BlockNode(n_items)
        
        # Statements
        case ReturnStatementNode(_) | ExpressionStatementNode(_) | GotoStatement(_) | NullStatementNode():
            return node
        case LabeledStatement(name, statement):
            return LabeledStatement(name, label_break_and_continue_statements(statement, labels))
        case CaseLabeledStatement(val, st, _, label):
            labels_copy = labels[:]
            while labels_copy and labels_copy[-1][1] != "switch":
                labels_copy.pop()
            if not labels_copy:
                raise SyntaxError
            return CaseLabeledStatement(val, label_break_and_continue_statements(st, labels), labels_copy[-1][0], label)
        case DefaultLabeledStatement(st, _, label):
            labels_copy = labels[:]
            while labels_copy and labels_copy[-1][1] != "switch":
                labels_copy.pop()
            if not labels_copy:
                raise SyntaxError
            return DefaultLabeledStatement(label_break_and_continue_statements(st, labels), labels_copy[-1][0], label)
        case IfStatementNode(cond, then_st, else_st):
            n_then_st = label_break_and_continue_statements(then_st, labels)
            n_else_st = label_break_and_continue_statements(else_st, labels) if else_st else None
            return IfStatementNode(cond, n_then_st, n_else_st)
        case CompoundStatement(block):
            n_block = label_break_and_continue_statements(block, labels)
            return CompoundStatement(n_block)
        case WhileStatementNode(cond, body, _):
            loop_label = get_label_name("while")
            n_body = label_break_and_continue_statements(body, labels + [(loop_label, "loop")])
            return WhileStatementNode(cond, n_body, loop_label)
        case DoWhileStatementNode(body, cond, label):
            loop_label = get_label_name("dowhile")
            n_body = label_break_and_continue_statements(body, labels + [(loop_label, "loop")])
            return DoWhileStatementNode(n_body, cond, loop_label)
        case ForStatementNode(init, cond, post, body, _):
            loop_label = get_label_name("for")
            n_body = label_break_and_continue_statements(body, labels + [(loop_label, "loop")])
            return ForStatementNode(init, cond, post, n_body, loop_label)
        case SwitchStatementNode(exp, body, cases, defaultCase, _):
            switch_label = get_label_name("switch")
            n_body = label_break_and_continue_statements(body, labels + [(switch_label, "switch")])
            return SwitchStatementNode(exp, n_body, cases, defaultCase, switch_label)
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
            return ContinueStatementNode(labels_copy[-1][0])
        case _:
            raise SyntaxError

def validate_prefix_and_postfix(node: AstNode, params: dict):
    match node:
        case PrefixExpressionNode(op, exp) | PostfixExpressionNode(op, exp):
            if not isinstance(exp, VariableExpressionNode):
                raise SyntaxError
        case _:
            pass

def validate_non_constant_cases(node: AstNode, params: dict):
    match node:
        case CaseLabeledStatement(val, statement, _, _):
            if not isinstance(val, ConstantExpressionNode):
                raise SyntaxError
        case _:
            pass

def assign_unique_labels_to_cases(node: AstNode, params: dict):
    match node:
        case CaseLabeledStatement(ConstantExpressionNode(val), _, switch_label, _) as c_node:
            c_node.label = get_label_name(f"{switch_label}.case{val}")
        case DefaultLabeledStatement(_, switch_label, _) as d_node:
            d_node.label = get_label_name(f"{switch_label}.default")
        case CaseLabeledStatement(_, _, _, _):
            raise SyntaxError("Wrong case format!")

def validate_case_uniqueness(node: AstNode, params: dict):
    cases_for_switches = params["cases"]
    defaults_for_switches = params["defaults"]
    match node:
        case CaseLabeledStatement(ConstantExpressionNode(val), statement, switch_label, _):
            if not switch_label in cases_for_switches:
                cases_for_switches[switch_label] = set()
            if val in cases_for_switches[switch_label]:
                raise SyntaxError("Duplicate case!")
            cases_for_switches[switch_label].add(val)
        case DefaultLabeledStatement(_, switch_label, _):
            if switch_label in defaults_for_switches:
                raise SyntaxError("Duplicate default label!")
            defaults_for_switches.add(switch_label)

def switch_add_cases_info(node: AstNode, params: dict):
    switches_dict = params["switches"]
    match node:
        case SwitchStatementNode(_, _, _, _, label):
            switches_dict[label] = node
        case DefaultLabeledStatement(_, switch_label, label):
            if not isinstance(switches_dict[switch_label], SwitchStatementNode):
                raise SyntaxError
            switch_node: SwitchStatementNode = switches_dict[switch_label]
            switch_node.defaultCase = label
        case CaseLabeledStatement(ConstantExpressionNode(val), _, switch_label, label):
            if not isinstance(switches_dict[switch_label], SwitchStatementNode):
                raise SyntaxError
            switch_node: SwitchStatementNode = switches_dict[switch_label]
            switch_node.cases.append((val, label))
        case _:
            pass

def validate(ast: AstNode) -> "AstNode":
    s1 = identifier_resolution(ast, 0, dict())
    label_map = dict()
    s2 = resolve_labels(s1, False, "", label_map)
    s3 = resolve_labels(s2, True, "", label_map)
    s4 = label_break_and_continue_statements(s3, [])
    traverse_ast(s4, {}, validate_prefix_and_postfix, None)
    traverse_ast(s4, {}, validate_non_constant_cases, None)
    traverse_ast(s4, {"cases": {}, "defaults": set()}, validate_case_uniqueness, None)
    traverse_ast(s4, {}, assign_unique_labels_to_cases, None)
    traverse_ast(s4, {"switches": {}}, switch_add_cases_info, None)

    return s4