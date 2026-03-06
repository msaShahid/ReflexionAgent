from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result of a tool execution."""
    tool_name: str
    success: bool
    output: str
    error: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_log_entry(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output_length": len(self.output),
            "error": self.error if self.error else None,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class ToolError(Exception):
    """Base exception for tool errors."""
    pass


class ToolExecutionError(ToolError):
    """Error during tool execution."""
    pass


class ToolValidationError(ToolError):
    """Error in tool input validation."""
    pass


class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self):
        self._last_result: Optional[ToolResult] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (unique identifier)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        pass
    
    @property
    def version(self) -> str:
        """Tool version."""
        return "1.0.0"
    
    @abstractmethod
    async def _execute(self, **kwargs: Any) -> str:
        """
        Internal execution method to be implemented by tools.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Tool output as string
            
        Raises:
            ToolExecutionError: If execution fails
        """
        pass
    
    def validate_input(self, **kwargs) -> None:
        """
        Validate input parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Raises:
            ToolValidationError: If validation fails
        """
        pass
    
    async def run(self, **kwargs: Any) -> ToolResult:
        """
        Run the tool with the given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with output or error
        """
        import time
        
        start_time = time.time()
        
        try:
            # Validate input
            self.validate_input(**kwargs)
            
            # Execute tool
            output = await self._execute(**kwargs)
            
            # Create successful result
            result = ToolResult(
                tool_name=self.name,
                success=True,
                output=output,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            
        except ToolValidationError as e:
            result = ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Validation error: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            result = ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Execution error: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        
        self._last_result = result
        return result
    
    def schema(self) -> Dict[str, Any]:
        """
        Get tool schema for LLM function calling.
        
        Returns:
            Dictionary with tool schema
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            }
        }
    
    def parameters_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool parameters.
        Override this to provide parameter specifications.
        
        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {},
        }
    
    @property
    def last_result(self) -> Optional[ToolResult]:
        """Get the last execution result."""
        return self._last_result


class AsyncBaseTool(BaseTool):
    """Base class for tools that need async initialization."""
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize tool resources."""
        pass
    
    async def cleanup(self):
        """Cleanup tool resources."""
        pass