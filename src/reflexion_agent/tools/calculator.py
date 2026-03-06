import ast
import operator
import re
from typing import Any, Dict, List, Union
from .base import BaseTool, ToolResult, ToolValidationError

# Whitelist of safe operations — never use eval() directly
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed functions
_SAFE_FUNCTIONS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
}


def _safe_eval(node: ast.AST) -> float:
    """Safely evaluate an AST node."""
    
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    
    elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](
            _safe_eval(node.left), 
            _safe_eval(node.right)
        )
    
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in _SAFE_FUNCTIONS:
            args = [_safe_eval(arg) for arg in node.args]
            return _SAFE_FUNCTIONS[node.func.id](*args)
    
    elif isinstance(node, ast.Compare):
        # Handle comparisons (e.g., 5 > 3)
        if len(node.ops) == 1 and len(node.comparators) == 1:
            left = _safe_eval(node.left)
            right = _safe_eval(node.comparators[0])
            
            op = node.ops[0]
            if isinstance(op, ast.Eq):
                return float(left == right)
            elif isinstance(op, ast.NotEq):
                return float(left != right)
            elif isinstance(op, ast.Lt):
                return float(left < right)
            elif isinstance(op, ast.LtE):
                return float(left <= right)
            elif isinstance(op, ast.Gt):
                return float(left > right)
            elif isinstance(op, ast.GtE):
                return float(left >= right)
    
    elif isinstance(node, ast.List):
        return [_safe_eval(elt) for elt in node.elts]
    
    elif isinstance(node, ast.Tuple):
        return tuple(_safe_eval(elt) for elt in node.elts)
    
    raise ToolValidationError(f"Unsafe or unsupported expression: {ast.dump(node)}")


def validate_expression(expression: str) -> bool:
    """
    Validate that expression contains only safe characters.
    
    Args:
        expression: Mathematical expression string
        
    Returns:
        True if safe, raises ToolValidationError if unsafe
    """
    # Check for dangerous patterns
    dangerous = ['__', 'import', 'exec', 'eval', 'open', 'file', 'globals', 'locals']
    for pattern in dangerous:
        if pattern in expression.lower():
            raise ToolValidationError(f"Expression contains forbidden pattern: {pattern}")
    
    # Only allow safe characters
    if not re.match(r'^[0-9\s\+\-\*\/\^\(\)\,\[\]\.\<\>\=]+$', expression):
        # Check if it's just numbers and basic operators
        allowed_chars = set('0123456789+-*/()., []<>=\n\t')
        if not all(c in allowed_chars for c in expression):
            raise ToolValidationError("Expression contains invalid characters")
    
    return True


class CalculatorTool(BaseTool):
    """Safe calculator tool for evaluating mathematical expressions."""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return """Evaluates safe arithmetic expressions. 
Supports: +, -, *, /, //, %, **, abs(), round(), min(), max(), sum()
Examples: '2 + 2', '(5 * 3) / 2', 'abs(-5)', 'max(10, 20, 30)'"""
    
    def validate_input(self, **kwargs) -> None:
        """Validate calculator input."""
        expression = kwargs.get('expression', '')
        
        if not isinstance(expression, str):
            raise ToolValidationError("Expression must be a string")
        
        if not expression.strip():
            raise ToolValidationError("Expression cannot be empty")
        
        # Safety validation
        validate_expression(expression)
    
    async def _execute(self, expression: str = "", **kwargs) -> str:
        """
        Execute calculator tool.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result as string
        """
        # Parse the expression
        tree = ast.parse(expression.strip(), mode="eval")
        
        # Evaluate safely
        result = _safe_eval(tree.body)
        
        # Format result
        if isinstance(result, float):
            # Handle floating point precision
            if result.is_integer():
                return str(int(result))
            else:
                # Limit decimal places
                return f"{result:.10f}".rstrip('0').rstrip('.')
        elif isinstance(result, (list, tuple)):
            return str(result)
        else:
            return str(result)
    
    def parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for calculator parameters."""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                    "examples": ["2 + 2", "(5 * 3) / 2", "abs(-5)"]
                }
            },
            "required": ["expression"]
        }