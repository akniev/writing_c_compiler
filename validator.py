from tacky import *
from parser import *

def resolve_variables(node: AstNode, variable_map: Dict[str, str]) -> AstNode:
    if node is None:
        return None

    match node:
        # Top level
        case ProgramNode(f_node):
            f_node_resolved = resolve_variables(f_node, variable_map)
            p_node = ProgramNode(f_node_resolved)
            return p_node
        case FunctionNode(name, block_items):
            block_items_resolved = []
            for bi in block_items:
                block_items_resolved.append(resolve_variables(bi, variable_map))
            f_node = FunctionNode(name, block_items_resolved)
            return f_node
        case DeclarationNode(name, exp):
            if name in variable_map:
                raise SyntaxError("Duplicate variable declaration!")
            name_resolved = get_temp_var_name(name)
            variable_map[name] = name_resolved
            d_node = DeclarationNode(name_resolved, resolve_variables(exp, variable_map))
            return d_node
        
        # Block item nodes
        case StatementBlockItemNode(statement):
            return StatementBlockItemNode(resolve_variables(statement, variable_map))
        case DeclarationBlockItemNode(declaration):
            return DeclarationBlockItemNode(resolve_variables(declaration, variable_map))
        
        # Statements
        case NullStatementNode():
            return node
        case ReturnStatementNode(exp):
            return ReturnStatementNode(resolve_variables(exp, variable_map))
        case ExpressionStatementNode(exp):
            return ExpressionStatementNode(resolve_variables(exp, variable_map))
        
        # Expressions
        case ConstantExpressionNode(_):
            return node
        case UnaryExpressionNode(unop, exp):
            return UnaryExpressionNode(unop, resolve_variables(exp, variable_map))
        case BinaryExpressionNode(binop, exp1, exp2):
            return BinaryExpressionNode(
                binop,
                resolve_variables(exp1, variable_map),
                resolve_variables(exp2, variable_map)
            )
        case VariableExpressionNode(name):
            if not name in variable_map:
                raise SyntaxError("Undeclared variable!")
            return VariableExpressionNode(variable_map[name])
        case AssignmentExpressionNode(lhs, rhs):
            if not isinstance(lhs, VariableExpressionNode):
                raise SyntaxError("Invalid lvalue!")
            return AssignmentExpressionNode(
                resolve_variables(lhs, variable_map),
                resolve_variables(rhs, variable_map)
            )
        case _:
            raise SyntaxError("Unknown AST node!")

def validate(ast: AstNode) -> "AstNode":
    return resolve_variables(ast, dict())