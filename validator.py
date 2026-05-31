from tacky import *
from parser import *
from validator_types import *


block_counter = 0


def get_block_id():
    global block_counter
    block_counter += 1
    return block_counter


def resolve_param(param: str, block_id: int, type: TypeNode, identifier_map: Dict[str, IdentifierMapEntry]) -> str:
    if param in identifier_map and identifier_map[param].block_id == block_id:
        raise SyntaxError("Duplicate variable declaration!")
    param_resolved = get_temp_var_name(param)
    identifier_map[param] = IdentifierMapEntry(param_resolved, block_id, type, False)
    return param_resolved

def identifier_resolution(node: AstNode, block_id: int, identifier_map: Dict[str, IdentifierMapEntry], linked_declarations: Dict[str, IdentifierMapEntry]) -> AstNode:
    if node is None:
        return None

    match node:
        # MARK: Top level
        case ProgramNode(decls):
            new_fun_decls = []
            for decl in decls:
                decl_resolved = identifier_resolution(decl, block_id, identifier_map, linked_declarations)
                new_fun_decls.append(decl_resolved)
            p_node = ProgramNode(new_fun_decls)
            return p_node
        
        # MARK: Declarations
        case FunctionDeclarationNode(name, params, body, fun_type, storage_class):
            if block_id != 0 and isinstance(storage_class, StaticStorageClass):
                raise SyntaxError("Wrong storage class")
            external_linkage = isinstance(storage_class, ExternStorageClass) or storage_class is None
            new_block_id = get_block_id()
            if name in identifier_map:
                prev_entry = identifier_map[name]
                if not prev_entry.has_linkage and prev_entry.block_id == 0:
                    external_linkage = False
                # if prev_entry.block_id == block_id and (not prev_entry.has_linkage):
                #     raise SyntaxError("Duplicate declaration!")
                if prev_entry.block_id == block_id and prev_entry.has_linkage != external_linkage:
                    raise SyntaxError("Conflicting function linkage!")

            map_entry = IdentifierMapEntry(name, block_id, fun_type, external_linkage)  
            if not external_linkage and name in linked_declarations:
                raise SyntaxError("Conflicting function linkage!")
            if external_linkage:
                linked_declarations[name] = map_entry
            identifier_map[name] = map_entry

            inner_map = identifier_map.copy()
            new_params = []
            for param in params:
                new_params.append(resolve_param(param, new_block_id, fun_type, inner_map))
            
            new_body: BlockNode|None = None
            if body is not None and block_id > 0:
                raise SyntaxError("Illegal function definition!")
            if body is not None:
                body_items = body.items
                new_body_items = []
                for body_item in body_items:
                    new_body_item = identifier_resolution(body_item, new_block_id, inner_map, linked_declarations)
                    new_body_items.append(new_body_item)
                new_body = BlockNode(new_body_items)
            return FunctionDeclarationNode(name, new_params, new_body, fun_type, storage_class)
        case VariableDeclarationNode(name, exp, var_type, storage_class):
            external_linkage = isinstance(storage_class, ExternStorageClass) or (storage_class is None and block_id == 0)
            if name in identifier_map:
                prev_entry = identifier_map[name]
                if prev_entry.block_id == 0 and isinstance(storage_class, ExternStorageClass) and not prev_entry.has_linkage:
                    external_linkage = False
                if prev_entry.block_id == block_id and external_linkage != prev_entry.has_linkage:
                     raise SyntaxError("Conflicting variable linkage!")
                
            
            if block_id == 0 and not external_linkage and name in linked_declarations:
                raise SyntaxError("Conflicting variable linkage!")
            if external_linkage:
                linked_declarations[name] = IdentifierMapEntry(name, block_id, var_type, external_linkage)

            if block_id == 0: # File scope variable
                identifier_map[name] = IdentifierMapEntry(name, block_id, var_type, external_linkage)
                return node

            # Local variables
            if name in identifier_map:
                prev_entry = identifier_map[name]
                if prev_entry.block_id == block_id:
                    if not (prev_entry.has_linkage and isinstance(storage_class, ExternStorageClass)):
                        raise SyntaxError("Conflicting local declarations")
            
            if isinstance(storage_class, ExternStorageClass):
                identifier_map[name] = IdentifierMapEntry(name, block_id, var_type, True)
                return node
            else:
                new_name = resolve_param(name, block_id, var_type, identifier_map)
                identifier_map[name] = IdentifierMapEntry(new_name, block_id, var_type, False)
                # if name in identifier_map and identifier_map[name].block_id == block_id:
                #     raise SyntaxError("Duplicate variable declaration!")
                # name_resolved = get_temp_var_name(name)
                # identifier_map[name] = IdentifierMapEntry(name_resolved, block_id, False)
                d_node = VariableDeclarationNode(new_name, identifier_resolution(exp, block_id, identifier_map, linked_declarations), var_type, storage_class)
                return d_node
        
        
        # MARK: Blocks
        case BlockNode(items):
            new_block_id = get_block_id()
            variable_map_copy = identifier_map.copy()
            items = [identifier_resolution(item, new_block_id, variable_map_copy, linked_declarations) for item in items]
            return BlockNode(items)
        
        case StatementBlockItemNode(statement):
            return StatementBlockItemNode(identifier_resolution(statement, block_id, identifier_map, linked_declarations))
        case DeclarationBlockItemNode(declaration):
            return DeclarationBlockItemNode(identifier_resolution(declaration, block_id, identifier_map, linked_declarations))
        
        # MARK: Statements
        case NullStatementNode():
            return node
        case ReturnStatementNode(exp):
            return ReturnStatementNode(identifier_resolution(exp, block_id, identifier_map, linked_declarations))
        case ExpressionStatementNode(exp):
            return ExpressionStatementNode(identifier_resolution(exp, block_id, identifier_map, linked_declarations))
        case IfStatementNode(cond, then_exp, else_exp):
            return IfStatementNode(
                identifier_resolution(cond, block_id, identifier_map, linked_declarations),
                identifier_resolution(then_exp, block_id, identifier_map, linked_declarations),
                identifier_resolution(else_exp, block_id, identifier_map, linked_declarations)
            )
        case GotoStatement(_):
            return node
        case LabeledStatement(name, statement):
            return LabeledStatement(name, identifier_resolution(statement, block_id, identifier_map, linked_declarations))
        case CompoundStatement(block):
            return CompoundStatement(identifier_resolution(block, block_id, identifier_map, linked_declarations))
        case BreakStatementNode(_):
            return node
        case ContinueStatementNode(_):
            return node
        
        case WhileStatementNode(cond, body, label):
            n_cond = identifier_resolution(cond, block_id, identifier_map, linked_declarations)
            n_body = identifier_resolution(body, block_id, identifier_map, linked_declarations)
            return WhileStatementNode(n_cond, n_body, label)
        case DoWhileStatementNode(body, cond, label):
            n_cond = identifier_resolution(cond, block_id, identifier_map, linked_declarations)
            n_body = identifier_resolution(body, block_id, identifier_map, linked_declarations)
            return DoWhileStatementNode(n_body, n_cond, label)
        case ForStatementNode(init, cond, post, body, label):
            new_block_id = get_block_id()
            new_variable_map = identifier_map.copy()
            n_init = identifier_resolution(init, new_block_id, new_variable_map, linked_declarations)
            n_cond = identifier_resolution(cond, new_block_id, new_variable_map, linked_declarations) if cond else None
            n_post = identifier_resolution(post, new_block_id, new_variable_map, linked_declarations) if post else None
            n_body = identifier_resolution(body, new_block_id, new_variable_map, linked_declarations)
            return ForStatementNode(n_init, n_cond, n_post, n_body, label)
        case ForInitDeclarationNode(decl):
            if decl.storage_class is not None:
                raise SyntaxError("Wrong variable declaration")
            n_decl = identifier_resolution(decl, block_id, identifier_map, linked_declarations)
            return ForInitDeclarationNode(n_decl)
        case ForInitExpressionNode(exp):
            n_exp = identifier_resolution(exp, block_id, identifier_map, linked_declarations)
            return ForInitExpressionNode(n_exp)
        case SwitchStatementNode(exp, body, cases, defaultCase, label):
            n_exp = identifier_resolution(exp, block_id, identifier_map, linked_declarations)
            n_body = identifier_resolution(body, block_id, identifier_map, linked_declarations)
            return SwitchStatementNode(n_exp, n_body, cases, defaultCase, label)
        case CaseLabeledStatement(val, st, switch_label, label):
            # val has to be a constant expression so we don't resolve variables for it
            n_st = identifier_resolution(st, block_id, identifier_map, linked_declarations)
            return CaseLabeledStatement(val, n_st, switch_label, label)
        case DefaultLabeledStatement(st, switch_label, label):
            n_st = identifier_resolution(st, block_id, identifier_map, linked_declarations)
            return DefaultLabeledStatement(n_st, switch_label, label)

        # MARK: Expressions
        case ConstIntExpressionNode(_, _):
            return node
        case ConstLongExpressionNode(_, _):
            return node
        case CastExpressionNode(exp_type, type_node, exp):
            n_exp = identifier_resolution(exp, block_id, identifier_map, linked_declarations)
            return CastExpressionNode(exp_type, type_node, n_exp)
        case UnaryExpressionNode(exp_type, unop, exp):
            n_exp = identifier_resolution(exp, block_id, identifier_map, linked_declarations)
            return UnaryExpressionNode(n_exp.exp_type, unop, n_exp)
        case BinaryExpressionNode(exp_type, binop, exp1, exp2):
            n_lhs = identifier_resolution(exp1, block_id, identifier_map, linked_declarations)
            n_rhs = identifier_resolution(exp2, block_id, identifier_map, linked_declarations)
            return BinaryExpressionNode(
                exp_type,
                binop,
                n_lhs,
                n_rhs
            )
        case PrefixExpressionNode(exp_type, op, exp):
            return PrefixExpressionNode(exp_type, op, identifier_resolution(exp, block_id, identifier_map, linked_declarations))
        case PostfixExpressionNode(exp_type, op, exp):
            return PostfixExpressionNode(exp_type, op, identifier_resolution(exp, block_id, identifier_map, linked_declarations))
        case VariableExpressionNode(exp_type, name):
            if not name in identifier_map:
                raise SyntaxError("Undeclared variable!")
            return VariableExpressionNode(exp_type, identifier_map[name].new_name)
        case AssignmentExpressionNode(exp_type, lhs, rhs):
            n_lhs = identifier_resolution(lhs, block_id, identifier_map, linked_declarations)
            n_rhs = identifier_resolution(rhs, block_id, identifier_map, linked_declarations)
            if not isinstance(n_lhs, VariableExpressionNode):
                raise SyntaxError("Invalid lvalue!")
            return AssignmentExpressionNode(
                exp_type,
                n_lhs,
                n_rhs
            )
        case CompoundAssignmentExpressionNode(exp_type, binop, lhs, rhs):
            n_lhs = identifier_resolution(lhs, block_id, identifier_map, linked_declarations)
            n_rhs = identifier_resolution(rhs, block_id, identifier_map, linked_declarations)
            if not isinstance(n_lhs, VariableExpressionNode):
                raise SyntaxError("Invalid lvalue!")
            return CompoundAssignmentExpressionNode(
                exp_type,
                binop,
                n_lhs,
                n_rhs
            )
        case ConditionalExpressionNode(exp_type, cond, true_exp, false_exp):
            n_cond = identifier_resolution(cond, block_id, identifier_map, linked_declarations)
            n_true_exp = identifier_resolution(true_exp, block_id, identifier_map, linked_declarations)
            n_false_exp = identifier_resolution(false_exp, block_id, identifier_map, linked_declarations)
            return ConditionalExpressionNode(
                exp_type,
                n_cond,
                n_true_exp,
                n_false_exp
            )
        case FunctionCallExpressionNode(exp_type, name, args, plt):
            if name in identifier_map:
                new_name = identifier_map[name].new_name
                new_args = []
                for arg in args:
                    new_args.append(identifier_resolution(arg, block_id, identifier_map, linked_declarations))
                return FunctionCallExpressionNode(exp_type, new_name, new_args, plt)
            else:
                raise SyntaxError("Undeclared function!")
        case _:
            raise SyntaxError("Unknown AST node!")

def get_minimum_common_type(type1: TypeNode, type2: TypeNode) -> TypeNode:
    if isinstance(type1, IntTypeNode) and isinstance(type2, IntTypeNode):
        return IntTypeNode()
    elif (isinstance(type1, IntTypeNode) and isinstance(type2, LongTypeNode)) or (isinstance(type1, LongTypeNode) and isinstance(type2, IntTypeNode)) or (isinstance(type1, LongTypeNode) and isinstance(type2, LongTypeNode)):
        return LongTypeNode()
    else:
        raise SyntaxError("Incompatible types!")


def resolve_labels(node: AstNode, update_goto: bool, func_prefix: str, label_map: Dict[str, str]) -> AstNode:
    if node is None:
        return None

    match node:
        # Top level
        case ProgramNode(function_declarations):
            resolved_function_declarations = []
            for fdecl in function_declarations:
                f_node_resolved = resolve_labels(fdecl, update_goto, func_prefix, label_map)
                resolved_function_declarations.append(f_node_resolved)
            p_node = ProgramNode(resolved_function_declarations)
            return p_node
        case FunctionDeclarationNode(name, params, block, fun_type, storage_class):
            return FunctionDeclarationNode(name, params, resolve_labels(block, update_goto, name, label_map), fun_type, storage_class)
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

type AstCallback = Callable[[AstNode, dict], None]
type AstProcessor = Callable[[AstNode, dict], Optional[AstNode]]

def process_ast(node: AstNode, params: dict, modify: AstProcessor, before: AstCallback, after: AstCallback) -> AstNode:
    if before:
        before(node, params)
    processed = modify(node, params)
    result = processed if processed is not None else node
    fs = vars(result)

    for f_name, f_val in list(fs.items()):
        if isinstance(f_val, AstNode):
            fs[f_name] = process_ast(f_val, params, modify, before, after)
        elif isinstance(f_val, (list, tuple, set)):
            for el in f_val:
                if isinstance(el, AstNode):
                    process_ast(el, params, modify, before, after)
    if after:
        after(node, params)
    return result

def traverse_ast(node: AstNode, params: dict, before: AstCallback, after: AstCallback):
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
        case FunctionDeclarationNode(name, params, body, fun_type, storage_class):
            n_body = label_break_and_continue_statements(body, labels) if body else None
            return FunctionDeclarationNode(name, params, n_body, fun_type, storage_class)
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
            if not isinstance(val, ConstIntExpressionNode):
                raise SyntaxError
        case _:
            pass

def assign_unique_labels_to_cases(node: AstNode, params: dict):
    match node:
        case CaseLabeledStatement(ConstIntExpressionNode(val), _, switch_label, _) as c_node:
            c_node.label = get_label_name(f"{switch_label}.case{val}")
        case DefaultLabeledStatement(_, switch_label, _) as d_node:
            d_node.label = get_label_name(f"{switch_label}.default")
        case CaseLabeledStatement(_, _, _, _):
            raise SyntaxError("Wrong case format!")

def validate_case_uniqueness(node: AstNode, params: dict):
    cases_for_switches = params["cases"]
    defaults_for_switches = params["defaults"]
    match node:
        case CaseLabeledStatement(ConstIntExpressionNode(val), statement, switch_label, _):
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
        case CaseLabeledStatement(ConstIntExpressionNode(val), _, switch_label, label):
            if not isinstance(switches_dict[switch_label], SwitchStatementNode):
                raise SyntaxError
            switch_node: SwitchStatementNode = switches_dict[switch_label]
            switch_node.cases.append((val, label))
        case _:
            pass


def get_fun_type(params):
    return f"Fun{len(params)}"


def typecheck_file_variable_declaration(node: VariableDeclarationNode, symbols: Dict[str, SymbolsTableItem], block_ids: List[int]) -> VariableDeclarationNode:
    match node.init:
        case ConstIntExpressionNode(value):
            initial_value = InitialValueInt(value)
        case ConstLongExpressionNode(value):
            initial_value = InitialValueLong(value)
        case None:
            if isinstance(node.storage_class, ExternStorageClass):
                initial_value = InitialValueNoInitializer()
            else:
                initial_value = InitialValueTentative()
        case _:
            raise SyntaxError("Non-constant initializer!")
    
    is_global = (not block_ids) or (not isinstance(node.storage_class, StaticStorageClass))

    if node.name in symbols:
        old_decl: SymbolsTableItem = symbols[node.name]
        if not isinstance(old_decl.type, (IntTypeNode, LongTypeNode)):
            raise SyntaxError("Function redeclared as variable")
        if node.var_type != old_decl.type:
            raise SyntaxError("Variable redeclared with different type!")
        if isinstance(node.storage_class, ExternStorageClass):
            is_global = old_decl.attrs.is_global
        elif old_decl.attrs.is_global != is_global:
            raise SyntaxError("Conflicting variable linkage!")
        
        if isinstance(old_decl.attrs.init, (InitialValueInt, InitialValueLong)):
            if isinstance(initial_value, (InitialValueInt, InitialValueLong)):
                raise SyntaxError("Coinflicting file scope variable definitions")
            else:
                initial_value = old_decl.attrs.init
        elif not isinstance(initial_value, (InitialValueInt, InitialValueLong, InitialValueTentative)):
            initial_value = InitialValueTentative()
    
    attrs = StaticAttr(initial_value, is_global)
    symbols[node.name] = SymbolsTableItem(node.name, node.var_type, attrs)
    return node

        
def typecheck_local_variable_declaration(node: VariableDeclarationNode, symbols: Dict[str, SymbolsTableItem], block_ids: List[int]) -> VariableDeclarationNode:
    if isinstance(node.storage_class, ExternStorageClass):
        if node.init is not None:
            raise SyntaxError("Initializer on local extern variable declaration!")
        
        if node.name in symbols:
            old_decl = symbols[node.name]
            if not isinstance(old_decl.type, (IntTypeNode, LongTypeNode)):
                raise SyntaxError("Function redeclared as variable")
            elif old_decl.type != node.var_type:
                raise SyntaxError("Variable redeclared with different type!")
            else:
                attrs = StaticAttr(InitialValueNoInitializer(), True)
                symbols[node.name] = SymbolsTableItem(node.name, node.var_type, attrs)
        elif isinstance(node.storage_class, StaticStorageClass):
            initial_value = None
            match node.init:
                case ConstIntExpressionNode(const_type, value):
                    initial_value = InitialValueInt(value)
                case ConstLongExpressionNode(const_type, value):
                    initial_value = InitialValueLong(value)
                case _:
                    raise SyntaxError("Non-constant initializer on local static variable")
            attrs = StaticAttr(initial_value, False)
            symbols[node.name] = SymbolsTableItem(node.name, node.var_type, attrs)
        elif isinstance(node.storage_class, ExternStorageClass):
            attrs = StaticAttr(InitialValueNoInitializer(), True)
            symbols[node.name] = SymbolsTableItem(node.name, node.var_type, attrs)
    else:
        if isinstance(node.storage_class, StaticStorageClass):
            initial_value = None
            match node.init:
                case ConstIntExpressionNode(const_type, value):
                    initial_value = InitialValueInt(value)
                case ConstLongExpressionNode(const_type, value):
                    initial_value = InitialValueLong(value)
                case None:
                    initial_value = InitialValueInt(0)
                case _:
                    raise SyntaxError("Non-constant initializer on local static variable")
            attrs = StaticAttr(initial_value, False)
            symbols[node.name] = SymbolsTableItem(node.name, node.var_type, attrs)
        else:
            symbols[node.name] = SymbolsTableItem(node.name, node.var_type, LocalAttr())
    if node.init is not None:
        n_init = typecheck_expression(node.init, symbols, block_ids)
        node = VariableDeclarationNode(node.name, n_init, node.var_type, node.storage_class)
    return node


def typecheck_function_declaration(node: FunctionDeclarationNode, symbols: Dict[str, SymbolsTableItem], block_ids: List[int]) -> FunctionDeclarationNode:
    fun_type = node.fun_type
    has_body = node.body is not None
    already_defined = False
    is_global = (not block_ids) or (not isinstance(node.storage_class, StaticStorageClass))
    if node.name in symbols:
        old_decl = symbols[node.name]
        if not isinstance(old_decl.type, FunTypeNode):
            raise SyntaxError("Variable redeclared as function")
        if node.fun_type.params != old_decl.type.params:
            raise SyntaxError("Function redeclared with different parameters!")
        if has_body and old_decl.attrs.is_defined:
            raise SyntaxError("Function redefinition!")
        already_defined = old_decl.attrs.is_defined
        if isinstance(node.storage_class, ExternStorageClass):
            is_global = old_decl.attrs.is_global
        elif old_decl.attrs.is_global != is_global:
            raise SyntaxError("Conflicting function linkage!")
    attrs = FunAttr(has_body or already_defined, is_global)
    symbols[node.name] = SymbolsTableItem(node.name, fun_type, attrs)
    
    if node.body is not None:
        # n_symbols = symbols.copy()
        for f_param, f_type in zip(node.params, node.fun_type.params):
            symbols[f_param] = SymbolsTableItem(f_param, f_type, False)
        n_body = typecheck_ast(node.body, symbols, block_ids)
        node = FunctionDeclarationNode(node.name, node.params, n_body, fun_type, node.storage_class)

    return node

    


def typecheck_ast(node: AstNode, symbols: Dict[str, SymbolsTableItem], block_ids: List[int]) -> AstNode:
    match node:
        case ProgramNode(decls):
            n_decls = []
            for decl in decls:
                n_decl = typecheck_ast(decl, symbols, block_ids)
                n_decls.append(n_decl)
            return ProgramNode(n_decls)
        case VariableDeclarationNode(name, init, var_type, storage_class):
            if not block_ids: # File scope
                return typecheck_file_variable_declaration(node, symbols, block_ids)
            else: # Local scope
                return typecheck_local_variable_declaration(node, symbols, block_ids)
        case FunctionDeclarationNode(name, params, body, fun_type, storage_class):
            return typecheck_function_declaration(node, symbols, block_ids)
        case StatementNode():
            return typecheck_statement(node, symbols, block_ids)
        case ExpressionNode():
            return typecheck_expression(node, symbols, block_ids)
        case ForInitDeclarationNode(decl):
            n_decl = typecheck_ast(decl, symbols, block_ids)
            return ForInitDeclarationNode(n_decl)
        case ForInitExpressionNode(exp):
            n_exp = typecheck_expression(exp, symbols, block_ids)
            return ForInitExpressionNode(n_exp)
        case BlockNode(items):
            new_block_ids = block_ids[:]
            new_block_ids.append(get_block_id())
            n_items = []
            for item in items:
                n_item = typecheck_ast(item, symbols, new_block_ids)
                n_items.append(n_item)
            return BlockNode(n_items)
        case StatementBlockItemNode(statement):
            n_statement = typecheck_statement(statement, symbols, block_ids)
            return StatementBlockItemNode(n_statement)
        case DeclarationBlockItemNode(declaration):
            n_declaration = typecheck_ast(declaration, symbols, block_ids)
            return DeclarationBlockItemNode(n_declaration)
        case _:
            raise SyntaxError("Unknown AST node!")


def typecheck_statement(node: StatementNode, symbols: Dict[str, SymbolsTableItem], block_ids: List[int]) -> StatementNode:
    match node:
        case ReturnStatementNode(exp):
            n_exp = typecheck_expression(exp, symbols, block_ids)
            return ReturnStatementNode(n_exp)
        case ExpressionStatementNode(exp):
            n_exp = typecheck_expression(exp, symbols, block_ids)
            return ExpressionStatementNode(n_exp)
        case IfStatementNode(cond, then_st, else_st):
            n_cond = typecheck_expression(cond, symbols, block_ids)
            n_then_st = typecheck_statement(then_st, symbols, block_ids)
            n_else_st = typecheck_statement(else_st, symbols, block_ids) if else_st else None
            return IfStatementNode(n_cond, n_then_st, n_else_st)
        case LabeledStatement(name, statement):
            n_statement = typecheck_statement(statement, symbols, block_ids)
            return LabeledStatement(name, n_statement)
        case CaseLabeledStatement(val, st, switch_label, label):
            n_st = typecheck_statement(st, symbols, block_ids)
            return CaseLabeledStatement(val, n_st, switch_label, label)
        case DefaultLabeledStatement(st, switch_label, label):
            n_st = typecheck_statement(st, symbols, block_ids)
            return DefaultLabeledStatement(n_st, switch_label, label)
        case GotoStatement(_):
            return node
        case CompoundStatement(block):
            n_block = typecheck_ast(block, symbols, block_ids)
            return CompoundStatement(n_block)
        case NullStatementNode():
            return node
        case BreakStatementNode(_):
            return node
        case ContinueStatementNode(_):
            return node
        case WhileStatementNode(cond, body, label):
            n_cond = typecheck_expression(cond, symbols, block_ids)
            n_body = typecheck_statement(body, symbols, block_ids)
            return WhileStatementNode(n_cond, n_body, label)
        case DoWhileStatementNode(body, cond, label):
            n_cond = typecheck_expression(cond, symbols, block_ids)
            n_body = typecheck_statement(body, symbols, block_ids)
            return DoWhileStatementNode(n_body, n_cond, label)
        case ForStatementNode(init, cond, post, body, label):
            n_init = typecheck_ast(init, symbols, block_ids) if init else None
            n_cond = typecheck_expression(cond, symbols, block_ids) if cond else None
            n_post = typecheck_ast(post, symbols, block_ids) if post else None
            n_body = typecheck_statement(body, symbols, block_ids)
            return ForStatementNode(n_init, n_cond, n_post, n_body, label)
        case SwitchStatementNode(exp, body, cases, defaultCase, label):
            n_exp = typecheck_expression(exp, symbols, block_ids)
            n_body = typecheck_statement(body, symbols, block_ids)
            n_cases = []
            for case in cases:
                n_cases.append(typecheck_statement(case, symbols, block_ids))
            return SwitchStatementNode(n_exp, n_body, n_cases, defaultCase, label)
        case _:
            raise SyntaxError("Unknown statement type!")

def typecheck_expression(node: ExpressionNode, symbols: Dict[str, SymbolsTableItem], block_ids: List[int]) -> ExpressionNode:
    match node:
        case ConstIntExpressionNode(_, _):
            return node
        case ConstLongExpressionNode(_, _):
            return node
        case UnaryExpressionNode(exp_type, unop, exp):
            n_exp = typecheck_expression(exp, symbols, block_ids)
            return UnaryExpressionNode(n_exp.exp_type, unop, n_exp)
        case PrefixExpressionNode(exp_type, op, exp):
            n_exp = typecheck_expression(exp, symbols, block_ids)
            return PrefixExpressionNode(exp_type, op, n_exp)
        case PostfixExpressionNode(exp_type, op, exp):
            n_exp = typecheck_expression(exp, symbols, block_ids)
            return PostfixExpressionNode(exp_type, op, n_exp)
        case BinaryExpressionNode(exp_type, binop, exp1, exp2):
            n_exp1 = typecheck_expression(exp1, symbols, block_ids)
            n_exp2 = typecheck_expression(exp2, symbols, block_ids)
            n_exp_type = get_minimum_common_type(n_exp1.exp_type, n_exp2.exp_type)
            return BinaryExpressionNode(n_exp_type, binop, n_exp1, n_exp2)
        case VariableExpressionNode(exp_type, name):
            if name not in symbols:
                raise SyntaxError("Undeclared variable!")
            if isinstance(symbols[name].type, FunTypeNode):
                raise SyntaxError("Function used as variable!")
            var_type = symbols[name].type
            return VariableExpressionNode(var_type, name)
        case CastExpressionNode(exp_type, type_node, exp):
            n_exp = typecheck_expression(exp, symbols, block_ids)
            return CastExpressionNode(type_node, type_node, n_exp)
        case AssignmentExpressionNode(exp_type, lhs, rhs):
            if not isinstance(lhs, VariableExpressionNode):
                raise SyntaxError("Invalid lvalue!")
            n_lhs = typecheck_expression(lhs, symbols, block_ids)
            n_rhs = typecheck_expression(rhs, symbols, block_ids)
            if get_minimum_common_type(n_lhs.exp_type, n_rhs.exp_type) != n_lhs.exp_type:
                raise SyntaxError("Type mismatch in assignment!")
            if n_lhs.exp_type == IntTypeNode() and n_rhs.exp_type != IntTypeNode():
                raise SyntaxError("Type mismatch in assignment!")
            return AssignmentExpressionNode(n_lhs.exp_type, n_lhs, n_rhs)
        case CompoundAssignmentExpressionNode(exp_type, binop, lhs, rhs):
            if not isinstance(lhs, VariableExpressionNode):
                raise SyntaxError("Invalid lvalue!")
            n_lhs = typecheck_expression(lhs, symbols, block_ids)
            n_rhs = typecheck_expression(rhs, symbols, block_ids)
            if get_minimum_common_type(n_lhs.exp_type, n_rhs.exp_type) != n_lhs.exp_type:
                raise SyntaxError("Type mismatch in assignment!")
            if n_lhs.exp_type == IntTypeNode() and n_rhs.exp_type != IntTypeNode():
                raise SyntaxError("Type mismatch in assignment!")
            return CompoundAssignmentExpressionNode(n_lhs.exp_type, binop, n_lhs, n_rhs)
        case ConditionalExpressionNode(exp_type, cond, true_exp, false_exp):
            n_cond = typecheck_expression(cond, symbols, block_ids)
            n_true_exp = typecheck_expression(true_exp, symbols, block_ids)
            n_false_exp = typecheck_expression(false_exp, symbols, block_ids)
            cond_type = get_minimum_common_type(n_true_exp.exp_type, n_false_exp.exp_type)
            return ConditionalExpressionNode(cond_type, n_cond, n_true_exp, n_false_exp)
        case FunctionCallExpressionNode(exp_type, name, args, plt):
            if name not in symbols:
                raise SyntaxError("Undeclared function!")
            n_args = []
            fun_type = symbols[name].type
            if not isinstance(fun_type, FunTypeNode):
                raise SyntaxError("Variable used as function!")
            if len(fun_type.params) != len(args):
                raise SyntaxError("Function called with wrong parameters!")
            for arg in args:
                n_args.append(typecheck_expression(arg, symbols, block_ids))
            return FunctionCallExpressionNode(fun_type.ret, name, n_args, plt)
        case _:
            raise SyntaxError("Unknown expression type!")

# def typecheck_ast(node: AstNode, symbols: Dict[str, SymbolsTableItem], block_ids: List[int]) -> AstNode:
#     match node:
#         case ProgramNode(declarations):
#             n_declarations = []
#             for decl in declarations:
#                 n_declarations.append(typecheck_ast(decl, symbols))
#             return ProgramNode(n_declarations)
#         case VariableDeclarationNode(name, init, var_type, storage_class):
#             var_type_str = ""
#             if isinstance(var_type, IntTypeNode):
#                 var_type = "int"
#             elif isinstance(var_type, LongTypeNode):
#                 var_type = "long"
#             else:
#                 raise SyntaxError("Unknown var type!")

#             block_id = block_ids[-1]
#             initial_value = None
#             if block_id == 0: # File scope
#                 match init:
#                     case ConstIntExpressionNode(value):
#                         initial_value = InitialValueInt(value)
#                     case ConstLongExpressionNode(value):
#                         initial_value = InitialValueLong(value)
#                     case None:
#                         if isinstance(storage_class, ExternStorageClass):
#                             initial_value = InitialValueNoInitializer()
#                         else:
#                             initial_value = InitialValueTentative()
#                     case _:
#                         raise SyntaxError("Non-constant initializer")
                
#                 is_global = not isinstance(storage_class, StaticStorageClass)

#                 if name in symbols:
#                     old_decl = symbols[name]
#                     if not old_decl.type in ["int", "long"]:
#                         raise SyntaxError("Function redeclared as variable")
#                     if isinstance(storage_class, ExternStorageClass):
#                         is_global = old_decl.attrs.is_global
#                     elif old_decl.attrs.is_global != is_global:
#                         raise SyntaxError("Conflicting variable linkage!")
                    
#                     if isinstance(old_decl.init, InitialValueInt) or isinstance(old_decl.init, InitialValueLong):
#                         if isinstance(initial_value, InitialValueInt) or isinstance(initial_value, InitialValueLong):
#                             raise SyntaxError("Conflicting file scope variable definitions")
#                         else:
#                             initial_value = old_decl.attrs.init
#                     elif not (isinstance(initial_value, InitialValueInt) or isinstance(initial_value, InitialValueLong)) and isinstance(old_decl.attrs.init, InitialValueTentative):
#                         initial_value = InitialValueTentative()

#                 attrs = StaticAttr(initial_value, is_global)
#                 symbols[name] = SymbolsTableItem(name, var_type_str, attrs)
#             else: # Local scope
#                 if isinstance(storage_class, ExternStorageClass):
#                     if init is not None:
#                         raise SyntaxError("Initializer on local extern variable declaration")

#                     if name in symbols:
#                         old_decl = symbols[name]
#                         if not old_decl.type in ["int", "long"]:
#                             raise SyntaxError("Function redeclared as variable")
#                     else:
#                         attrs = StaticAttr(InitialValueNoInitializer(), True)
#                         symbols[name] = SymbolsTableItem(name, var_type_str, attrs)
#                 elif isinstance(storage_class, StaticStorageClass):
#                     initial_value = None
#                     match init:
#                         case ConstIntExpressionNode(value):
#                             initial_value = InitialValueInt(value)
#                         case ConstLongExpressionNode(value):
#                             initial_value = InitialValueLong(value)
#                         case None:
#                             initial_value = InitialValueInt(0)
#                         case _:
#                             raise SyntaxError("Non-constant initializer on local static variable")
#                     attrs = StaticAttr(initial_value, False)
#                     symbols[name] = SymbolsTableItem(name, var_type_str, attrs)
#                 else:
#                     symbols[name] = SymbolsTableItem(name, var_type_str, LocalAttr())
#         case FunctionDeclarationNode()


# def typecheck_ast(node: AstNode, symbols: Dict[str, SymbolsTableItem]):
#     def process(node: AstNode, params: dict):
#         return node
    
#     def before(node: AstNode, params: dict):
#         symbols: Dict[str, SymbolsTableItem] = params["symbols"]
#         block_ids: List[int] = params["block_ids"]
#         match node:
#             case BlockNode(_):
#                 new_block_id = get_block_id()
#                 block_ids.append(new_block_id)
#             case VariableDeclarationNode(name, init, storage_class):
#                 block_id = block_ids[-1]
#                 initial_value = None
#                 if block_id == 0: # File scope
#                     match init:
#                         case ConstIntExpressionNode(value):
#                             initial_value = InitialValueInt(value)
#                         case None:
#                             if isinstance(storage_class, ExternStorageClass):
#                                 initial_value = InitialValueNoInitializer()
#                             else:
#                                 initial_value = InitialValueTentative()
#                         case _:
#                             raise SyntaxError("Non-constant initializer!")
                    
#                     is_global = not isinstance(storage_class, StaticStorageClass)

#                     if name in symbols:
#                         old_decl = symbols[name]

#                         if old_decl.type != "Int":
#                             raise SyntaxError("Function redeclared as variable")
#                         if isinstance(storage_class, ExternStorageClass):
#                             is_global = old_decl.attrs.is_global
#                         elif old_decl.attrs.is_global != is_global:
#                             raise SyntaxError("Conflicting variable linkage")
                        
#                         if isinstance(old_decl.attrs.init, InitialValueInt):
#                             if isinstance(initial_value, InitialValueInt):
#                                 raise SyntaxError("Conflicting file scope variable definitions")
#                             else:
#                                 initial_value = old_decl.attrs.init
#                         elif not isinstance(initial_value, InitialValueInt) and isinstance(old_decl.attrs.init, InitialValueTentative):
#                             initial_value = InitialValueTentative()
                    
#                     attrs = StaticAttr(initial_value, is_global)
#                     symbols[name] = SymbolsTableItem(name, "Int", attrs)
#                 else: # Local scope
#                     if isinstance(storage_class, ExternStorageClass):
#                         if init is not None:
#                             raise SyntaxError("Initializer on local extern variable declaration")

#                         if name in symbols:
#                             old_decl = symbols[name]
#                             if old_decl.type != "Int":
#                                 raise SyntaxError("Function redeclared as variable")
#                         else:
#                             attrs = StaticAttr(InitialValueNoInitializer(), True)
#                             symbols[name] = SymbolsTableItem(name, "Int", attrs)
#                     elif isinstance(storage_class, StaticStorageClass):
#                         initial_value = None
#                         match init:
#                             case ConstIntExpressionNode(value):
#                                 initial_value = InitialValueInt(value)
#                             case None:
#                                 initial_value = InitialValueInt(0)
#                             case _:
#                                 raise SyntaxError("Non-constant initializer on local static variable")
                        
#                         attrs = StaticAttr(initial_value, False)
#                         symbols[name] = SymbolsTableItem(name, "Int", attrs)
#                     else:
#                         symbols[name] = SymbolsTableItem(name, "Int", LocalAttr())
#                         # if init is not None:
#                         #     typecheck_ast(init, symbols)
                
#                 # symbols[name] = SymbolsTableItem(name, "Int", False)
#             case FunctionDeclarationNode(name, f_params, body, storage_class):
#                 fun_type = get_fun_type(f_params)
#                 has_body = body is not None
#                 already_defined = False
#                 is_global = not isinstance(storage_class, StaticStorageClass)

#                 if name in symbols:
#                     old_decl = symbols[name]
#                     if not isinstance(old_decl.attrs, FunAttr):
#                         raise SyntaxError("Wrong attribute type")
#                     old_attrs: FunAttr = old_decl.attrs
#                     if old_decl.type != fun_type:
#                         raise SyntaxError("Incompatible function declarations")
#                     already_defined = old_attrs.is_defined
#                     if already_defined and has_body:
#                         raise SyntaxError("Function is defined more than once")
#                     if old_attrs.is_global and isinstance(storage_class, StaticStorageClass):
#                         raise SyntaxError("Static function declaration follows non-static")
#                     is_global = old_attrs.is_global

#                 attrs = FunAttr((already_defined or has_body), is_global)
#                 symbols[name] = SymbolsTableItem(name, fun_type, attrs)

#                 if has_body:
#                     for f_param in f_params:
#                         symbols[f_param] = SymbolsTableItem(f_param, "Int", False)
#             case FunctionCallExpressionNode(name, args):
#                 f_type = symbols[name].type
#                 if f_type == "Int":
#                     raise SyntaxError("Variable used as function name")
#                 if f_type != get_fun_type(args):
#                     raise SyntaxError("Function calledc with the wrong number of arguments")
#             case VariableExpressionNode(exp_type, name):
#                 if symbols[name].type != "Int":
#                     raise SyntaxError("Function mame used as variable")

#     def after(node: AstNode, params: dict):
#         block_ids: List[int] = params["block_ids"]
#         match node:
#             case BlockNode(_):
#                 block_ids.pop()
#         pass

#     return process_ast(node, {"symbols": symbols, "block_ids": [0]}, process, before, after)

def save_defined_functions(node: AstNode, params: dict):
    defined_functions = params["defined_functions"]
    match node:
        case FunctionDeclarationNode(name, _, body):
            if body is not None:
                defined_functions.add(name)

def set_plt_flat_for_defined_functions(node: AstNode, params: dict):
    defined_functions = params["defined_functions"]
    match node:
        case FunctionCallExpressionNode(type, name, args, _):
            plt = not (name in defined_functions)
            return FunctionCallExpressionNode(type, name, args, plt)
        case _:
            return node


def validate(ast: AstNode) -> "AstNode":
    s1 = identifier_resolution(ast, 0, dict(), dict())
    label_map = dict()
    s2 = resolve_labels(s1, False, "", label_map)
    s3 = resolve_labels(s2, True, "", label_map)
    s4 = label_break_and_continue_statements(s3, [])
    traverse_ast(s4, {}, validate_prefix_and_postfix, None)
    traverse_ast(s4, {}, validate_non_constant_cases, None)
    traverse_ast(s4, {"cases": {}, "defaults": set()}, validate_case_uniqueness, None)
    traverse_ast(s4, {}, assign_unique_labels_to_cases, None)
    traverse_ast(s4, {"switches": {}}, switch_add_cases_info, None)

    defined_functions = set()
    traverse_ast(s4, { "defined_functions": defined_functions }, save_defined_functions, None)
    s5 = process_ast(s4, {"defined_functions": defined_functions}, set_plt_flat_for_defined_functions, None, None)

    symbols = dict()
    typecheck_ast(s5, symbols, [])
    return s4, symbols