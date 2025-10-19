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
            f_node = FunctionNode(name, block_items_resolved)
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
        case LabeledStatement(_):
            return node
        case CompoundStatement(block):
            return CompoundStatement(resolve_variables(block, block_id, variable_map))
        
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
        case FunctionNode(name, BlockNode(block_items)):
            block_items_resolved = []
            for bi in block_items:
                block_items_resolved.append(resolve_labels(bi, update_goto, func_prefix, label_map))
            f_node = FunctionNode(name, block_items_resolved)
            return f_node
        
        # Block item nodes
        case StatementBlockItemNode(statement):
            return StatementBlockItemNode(resolve_labels(statement, update_goto, func_prefix, label_map))
        
        case IfStatementNode(cond, then_exp, else_exp):
            return IfStatementNode(
                cond,
                resolve_labels(then_exp, update_goto, func_prefix, label_map),
                resolve_labels(else_exp, update_goto, func_prefix, label_map)
            )
        case GotoStatement(label):
            if not update_goto:
                return node
            fname = f"{func_prefix}.{label}"
            if fname not in label_map:
                raise SyntaxError("Unknown label!")
            return GotoStatement(label_map[fname])
        case LabeledStatement(name):
            if update_goto:
                return node
            fname = f"{func_prefix}.{name}"
            if fname in label_map:
                raise SyntaxError("Duplicate labels!")
            new_name = get_label_name(fname)
            label_map[fname] = new_name
            return LabeledStatement(new_name)
        case _:
            return node

def check_labels_have_statements(node: AstNode):
    match node:
        case ProgramNode(function):
            return check_labels_have_statements(function)
        case FunctionNode(_, block_items):
            for i in range(len(block_items)):
                block_item = block_items[i]
                next_block_item = block_items[i + 1] if i < len(block_items) - 1 else None
                match (block_item, next_block_item):
                    case StatementBlockItemNode(LabeledStatement(_)), None:
                        raise SyntaxError("No statements are labeled")
                    case StatementBlockItemNode(LabeledStatement(_)), DeclarationBlockItemNode(_):
                        raise SyntaxError("no labels for declarations")
                    case _:
                        pass
        case _:
            raise SyntaxError

def validate(ast: AstNode) -> "AstNode":
    s1 = resolve_variables(ast, 0, dict())
    label_map = dict()
    s2 = resolve_labels(s1, False, "", label_map)
    s3 = resolve_labels(s2, True, "", label_map)
    check_labels_have_statements(s3)
    return s3