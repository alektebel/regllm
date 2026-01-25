#!/usr/bin/env python3
"""
Tool Executor for Agent Framework

Handles execution of registered tools with validation, error handling,
and result formatting.
"""

import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime

from .tool_registry import Tool, ToolRegistry, default_registry

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS

    def format_for_llm(self) -> str:
        """Format result for LLM consumption."""
        if self.is_success:
            if isinstance(self.result, (dict, list)):
                return json.dumps(self.result, ensure_ascii=False, indent=2)
            return str(self.result)
        else:
            return f"Error ejecutando {self.tool_name}: {self.error}"


class ToolExecutor:
    """Executes tools from the registry with validation and error handling."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        """Initialize executor with a tool registry.

        Args:
            registry: Tool registry to use. Defaults to global registry.
        """
        self.registry = registry or default_registry
        self.execution_history: List[ToolResult] = []
        self.max_history_size = 100

    def validate_parameters(
        self, tool: Tool, parameters: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate parameters against tool's JSON schema.

        Args:
            tool: The tool to validate parameters for.
            parameters: Parameters to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        schema = tool.parameters
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required parameters
        for param_name in required:
            if param_name not in parameters:
                return False, f"Parámetro requerido faltante: {param_name}"

        # Type validation for provided parameters
        for param_name, param_value in parameters.items():
            if param_name not in properties:
                continue

            param_schema = properties[param_name]
            expected_type = param_schema.get("type")

            if expected_type and not self._check_type(param_value, expected_type):
                return False, (
                    f"Tipo incorrecto para {param_name}: "
                    f"esperado {expected_type}, recibido {type(param_value).__name__}"
                )

            # Enum validation
            if "enum" in param_schema and param_value not in param_schema["enum"]:
                return False, (
                    f"Valor inválido para {param_name}: {param_value}. "
                    f"Valores permitidos: {param_schema['enum']}"
                )

        return True, None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        if expected_type not in type_mapping:
            return True  # Unknown type, allow

        expected_python_type = type_mapping[expected_type]
        return isinstance(value, expected_python_type)

    def execute(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> ToolResult:
        """Execute a tool by name with given parameters.

        Args:
            tool_name: Name of the tool to execute.
            parameters: Parameters to pass to the tool.
            validate: Whether to validate parameters before execution.

        Returns:
            ToolResult with execution outcome.
        """
        parameters = parameters or {}
        start_time = datetime.now()

        # Get tool from registry
        tool = self.registry.get(tool_name)
        if not tool:
            result = ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.NOT_FOUND,
                error=f"Herramienta no encontrada: {tool_name}",
            )
            self._add_to_history(result)
            return result

        # Validate parameters
        if validate:
            is_valid, error_msg = self.validate_parameters(tool, parameters)
            if not is_valid:
                result = ToolResult(
                    tool_name=tool_name,
                    status=ExecutionStatus.VALIDATION_ERROR,
                    error=error_msg,
                )
                self._add_to_history(result)
                return result

        # Execute tool
        try:
            execution_result = tool.function(**parameters)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            result = ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.SUCCESS,
                result=execution_result,
                execution_time_ms=execution_time,
            )

            logger.info(
                f"Tool {tool_name} executed successfully in {execution_time:.2f}ms"
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_trace = traceback.format_exc()

            result = ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.ERROR,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time,
            )

            logger.error(f"Tool {tool_name} failed: {e}\n{error_trace}")

        self._add_to_history(result)
        return result

    def execute_multiple(
        self,
        tool_calls: List[Dict[str, Any]],
        stop_on_error: bool = False,
    ) -> List[ToolResult]:
        """Execute multiple tools in sequence.

        Args:
            tool_calls: List of dicts with 'name' and 'parameters' keys.
            stop_on_error: Whether to stop execution on first error.

        Returns:
            List of ToolResults for each execution.
        """
        results = []

        for call in tool_calls:
            tool_name = call.get("name")
            parameters = call.get("parameters", {})

            result = self.execute(tool_name, parameters)
            results.append(result)

            if stop_on_error and not result.is_success:
                logger.warning(f"Stopping execution due to error in {tool_name}")
                break

        return results

    def parse_and_execute(self, tool_call_json: str) -> ToolResult:
        """Parse a JSON tool call and execute it.

        Args:
            tool_call_json: JSON string with tool call specification.

        Returns:
            ToolResult with execution outcome.
        """
        try:
            call_data = json.loads(tool_call_json)
        except json.JSONDecodeError as e:
            return ToolResult(
                tool_name="unknown",
                status=ExecutionStatus.VALIDATION_ERROR,
                error=f"JSON inválido: {e}",
            )

        tool_name = call_data.get("name") or call_data.get("tool_name")
        parameters = call_data.get("parameters") or call_data.get("arguments", {})

        if not tool_name:
            return ToolResult(
                tool_name="unknown",
                status=ExecutionStatus.VALIDATION_ERROR,
                error="Nombre de herramienta no especificado",
            )

        return self.execute(tool_name, parameters)

    def _add_to_history(self, result: ToolResult):
        """Add result to execution history with size limit."""
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]

    def get_history(
        self,
        tool_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 10,
    ) -> List[ToolResult]:
        """Get execution history with optional filters.

        Args:
            tool_name: Filter by tool name.
            status: Filter by execution status.
            limit: Maximum results to return.

        Returns:
            Filtered list of ToolResults.
        """
        results = self.execution_history

        if tool_name:
            results = [r for r in results if r.tool_name == tool_name]

        if status:
            results = [r for r in results if r.status == status]

        return results[-limit:]

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.registry.list_tools()]

    def get_tools_description(self) -> str:
        """Get formatted description of all available tools."""
        return self.registry.get_tools_prompt()


# Global executor instance
default_executor = ToolExecutor()


def main():
    """Demo of tool executor."""
    from .tool_registry import Tool

    # Create test registry and executor
    registry = ToolRegistry()
    executor = ToolExecutor(registry)

    # Register a sample tool
    def calculate_rwa(pd: float, lgd: float, ead: float) -> Dict[str, Any]:
        """Calculate Risk-Weighted Assets."""
        if not (0 <= pd <= 1):
            raise ValueError("PD must be between 0 and 1")
        if not (0 <= lgd <= 1):
            raise ValueError("LGD must be between 0 and 1")

        # Simplified RWA calculation
        k = lgd * pd * 12.5
        rwa = k * ead

        return {
            "pd": pd,
            "lgd": lgd,
            "ead": ead,
            "k": k,
            "rwa": rwa,
        }

    tool = Tool(
        name="calculate_rwa",
        description="Calcula los activos ponderados por riesgo (RWA)",
        parameters={
            "type": "object",
            "properties": {
                "pd": {
                    "type": "number",
                    "description": "Probabilidad de impago (0-1)",
                },
                "lgd": {
                    "type": "number",
                    "description": "Pérdida dado el impago (0-1)",
                },
                "ead": {
                    "type": "number",
                    "description": "Exposición en caso de impago",
                },
            },
            "required": ["pd", "lgd", "ead"],
        },
        function=calculate_rwa,
        category="calculation",
    )

    registry.register(tool)

    # Test execution
    print("Testing tool execution:")
    print("-" * 40)

    # Successful execution
    result = executor.execute("calculate_rwa", {"pd": 0.02, "lgd": 0.45, "ead": 1000000})
    print(f"Success: {result.is_success}")
    print(f"Result: {result.result}")
    print()

    # Validation error (missing parameter)
    result = executor.execute("calculate_rwa", {"pd": 0.02, "lgd": 0.45})
    print(f"Success: {result.is_success}")
    print(f"Error: {result.error}")
    print()

    # Tool not found
    result = executor.execute("unknown_tool", {})
    print(f"Success: {result.is_success}")
    print(f"Error: {result.error}")


if __name__ == "__main__":
    main()
