#!/usr/bin/env python3
"""
Agent Loop for RegLLM

Provides an agent that can reason about regulatory tasks, select tools,
and execute multi-step workflows.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime

from .tool_registry import Tool, ToolRegistry, default_registry
from .tool_executor import ToolExecutor, ToolResult, ExecutionStatus

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution state."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOL = "executing_tool"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message in agent conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_call: Optional[Dict] = None
    tool_result: Optional[ToolResult] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.tool_call:
            d["tool_call"] = self.tool_call
        if self.tool_result:
            d["tool_result"] = self.tool_result.to_dict()
        return d


@dataclass
class AgentStep:
    """A single step in agent execution."""
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    is_final: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "is_final": self.is_final,
        }


class RegulationAgent:
    """Agent specialized in regulatory analysis and methodology consistency checking.

    This agent can:
    - Answer questions about banking regulations
    - Compare methodologies
    - Analyze code implementations
    - Check consistency between methodology documents and code
    """

    SYSTEM_PROMPT = """Eres un agente experto en regulación bancaria española y europea.

Tu objetivo es ayudar a analizar documentos de metodología y verificar que las implementaciones de código sean consistentes con las especificaciones de la metodología.

Tienes acceso a las siguientes herramientas:

{tools_description}

Para usar una herramienta, responde en este formato:

PENSAMIENTO: [Tu razonamiento sobre qué hacer a continuación]
ACCIÓN: [nombre_de_herramienta]
ENTRADA: [parámetros en JSON]

Si tienes la respuesta final, responde:

PENSAMIENTO: [Tu razonamiento final]
RESPUESTA_FINAL: [Tu respuesta completa]

Reglas:
1. Siempre explica tu razonamiento antes de actuar
2. Usa las herramientas disponibles para obtener información
3. Sé preciso y cita fuentes cuando sea posible
4. Responde siempre en español
5. Si no puedes completar una tarea, explica por qué"""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        llm_client: Optional[Callable] = None,
        max_steps: int = 10,
        verbose: bool = True,
    ):
        """Initialize the regulation agent.

        Args:
            registry: Tool registry to use.
            llm_client: Callable that takes messages and returns response.
            max_steps: Maximum reasoning steps before stopping.
            verbose: Whether to log detailed execution info.
        """
        self.registry = registry or default_registry
        self.executor = ToolExecutor(self.registry)
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.verbose = verbose

        self.state = AgentState.IDLE
        self.conversation: List[AgentMessage] = []
        self.steps: List[AgentStep] = []

    def _get_system_prompt(self) -> str:
        """Generate system prompt with available tools."""
        tools_desc = self.registry.get_tools_prompt()
        return self.SYSTEM_PROMPT.format(tools_description=tools_desc)

    def _parse_response(self, response: str) -> AgentStep:
        """Parse agent response into structured step.

        Args:
            response: Raw response from LLM.

        Returns:
            Parsed AgentStep.
        """
        step = AgentStep(
            step_number=len(self.steps) + 1,
            thought="",
        )

        # Extract thought
        thought_match = re.search(
            r"PENSAMIENTO:\s*(.+?)(?=ACCIÓN:|RESPUESTA_FINAL:|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            step.thought = thought_match.group(1).strip()

        # Check for final answer
        final_match = re.search(
            r"RESPUESTA_FINAL:\s*(.+?)$",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match:
            step.is_final = True
            step.observation = final_match.group(1).strip()
            return step

        # Extract action
        action_match = re.search(
            r"ACCIÓN:\s*(\w+)",
            response,
            re.IGNORECASE,
        )
        if action_match:
            step.action = action_match.group(1).strip()

        # Extract action input
        input_match = re.search(
            r"ENTRADA:\s*(\{.+?\}|\[.+?\])",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if input_match:
            try:
                step.action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                step.action_input = {"raw": input_match.group(1)}

        return step

    def _execute_step(self, step: AgentStep) -> str:
        """Execute a step's action and return observation.

        Args:
            step: The step to execute.

        Returns:
            Observation string from tool execution.
        """
        if not step.action:
            return "No se especificó ninguna acción"

        self.state = AgentState.EXECUTING_TOOL

        result = self.executor.execute(
            step.action,
            step.action_input or {},
        )

        return result.format_for_llm()

    def _format_observation(self, observation: str) -> str:
        """Format observation for the conversation."""
        return f"OBSERVACIÓN: {observation}"

    def run(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the agent on a query.

        Args:
            query: User's question or task.
            context: Optional additional context.

        Returns:
            Dict with final answer, steps taken, and execution info.
        """
        self.state = AgentState.THINKING
        self.steps = []

        # Build initial messages
        system_msg = AgentMessage(
            role="system",
            content=self._get_system_prompt(),
        )
        self.conversation = [system_msg]

        # Add context if provided
        user_content = query
        if context:
            user_content = f"Contexto:\n{context}\n\nPregunta: {query}"

        user_msg = AgentMessage(role="user", content=user_content)
        self.conversation.append(user_msg)

        final_answer = None
        error = None

        for i in range(self.max_steps):
            if self.verbose:
                logger.info(f"Agent step {i + 1}/{self.max_steps}")

            # Get LLM response
            try:
                response = self._get_llm_response()
            except Exception as e:
                error = f"Error obteniendo respuesta del modelo: {e}"
                self.state = AgentState.ERROR
                break

            # Parse response
            step = self._parse_response(response)
            self.steps.append(step)

            if self.verbose:
                logger.info(f"Thought: {step.thought[:100]}...")
                if step.action:
                    logger.info(f"Action: {step.action}")

            # Check if we have final answer
            if step.is_final:
                final_answer = step.observation
                self.state = AgentState.COMPLETED
                break

            # Execute action if specified
            if step.action:
                observation = self._execute_step(step)
                step.observation = observation

                # Add observation to conversation
                obs_msg = AgentMessage(
                    role="assistant",
                    content=response,
                )
                self.conversation.append(obs_msg)

                obs_feedback = AgentMessage(
                    role="user",
                    content=self._format_observation(observation),
                )
                self.conversation.append(obs_feedback)

            self.state = AgentState.THINKING

        # If we reached max steps without answer
        if not final_answer and not error:
            final_answer = self._summarize_steps()
            self.state = AgentState.COMPLETED

        return {
            "query": query,
            "answer": final_answer,
            "steps": [s.to_dict() for s in self.steps],
            "state": self.state.value,
            "error": error,
            "total_steps": len(self.steps),
        }

    def _get_llm_response(self) -> str:
        """Get response from LLM.

        Returns:
            LLM response string.
        """
        if self.llm_client:
            messages = [
                {"role": m.role, "content": m.content}
                for m in self.conversation
            ]
            return self.llm_client(messages)
        else:
            # Return a mock response for testing without LLM
            return self._mock_response()

    def _mock_response(self) -> str:
        """Generate mock response for testing."""
        if len(self.steps) == 0:
            # First step - try to use a tool
            return """PENSAMIENTO: Necesito analizar la consulta del usuario y determinar qué herramientas usar.

ACCIÓN: search_regulations
ENTRADA: {"query": "metodología IRB", "limit": 3}"""

        # After one step, give final answer
        return """PENSAMIENTO: He recopilado la información necesaria para responder.

RESPUESTA_FINAL: Basándome en mi análisis, puedo proporcionar la siguiente respuesta sobre la consulta."""

    def _summarize_steps(self) -> str:
        """Summarize steps taken when max steps reached."""
        summary_parts = [
            "Se alcanzó el límite máximo de pasos. Resumen de acciones realizadas:\n"
        ]

        for step in self.steps:
            summary_parts.append(f"\nPaso {step.step_number}:")
            summary_parts.append(f"  Pensamiento: {step.thought[:200]}...")
            if step.action:
                summary_parts.append(f"  Acción: {step.action}")
            if step.observation:
                summary_parts.append(f"  Resultado: {step.observation[:200]}...")

        return "\n".join(summary_parts)

    def reset(self):
        """Reset agent state for new conversation."""
        self.state = AgentState.IDLE
        self.conversation = []
        self.steps = []
        self.executor.execution_history = []


class MethodologyConsistencyAgent(RegulationAgent):
    """Specialized agent for checking methodology-code consistency.

    This agent is designed to:
    1. Read and parse methodology documents
    2. Analyze code implementations
    3. Check if code correctly implements the methodology
    4. Report inconsistencies with specific references
    """

    SYSTEM_PROMPT = """Eres un agente especializado en verificar la consistencia entre documentos de metodología regulatoria y su implementación en código.

Tu objetivo principal es:
1. Leer y entender documentos de metodología (fórmulas, requisitos, parámetros)
2. Analizar código que implementa esa metodología
3. Verificar que el código sea consistente con la metodología
4. Reportar cualquier inconsistencia encontrada

Tienes acceso a las siguientes herramientas:

{tools_description}

Para usar una herramienta, responde en este formato:

PENSAMIENTO: [Tu razonamiento sobre qué verificar]
ACCIÓN: [nombre_de_herramienta]
ENTRADA: [parámetros en JSON]

Cuando tengas tu análisis completo, responde:

PENSAMIENTO: [Tu razonamiento final]
RESPUESTA_FINAL: [Tu reporte de consistencia con el siguiente formato]

## Reporte de Consistencia

### Metodología Analizada
[Nombre y descripción]

### Código Analizado
[Archivos y funciones revisadas]

### Resultados
- **Consistentes**: [Lista de elementos que coinciden]
- **Inconsistencias**: [Lista de problemas encontrados con referencias específicas]
- **Advertencias**: [Posibles problemas o áreas de mejora]

### Conclusión
[Resumen general de la consistencia]

Reglas:
1. Sé exhaustivo en el análisis
2. Cita líneas específicas del código y secciones de la metodología
3. Distingue entre errores críticos y advertencias menores
4. Proporciona recomendaciones de corrección cuando encuentres inconsistencias
5. Responde siempre en español"""

    def check_consistency(
        self,
        methodology_path: str,
        code_path: str,
        aspects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check consistency between methodology document and code.

        Args:
            methodology_path: Path to methodology document.
            code_path: Path to code file or directory.
            aspects: Specific aspects to check (optional).

        Returns:
            Consistency report with findings.
        """
        context = f"""Documento de metodología: {methodology_path}
Código a verificar: {code_path}"""

        if aspects:
            context += f"\nAspectos específicos a verificar: {', '.join(aspects)}"

        query = """Analiza el documento de metodología y el código proporcionado.
Verifica que el código implemente correctamente todos los aspectos de la metodología.
Reporta cualquier inconsistencia encontrada."""

        return self.run(query, context=context)


def main():
    """Demo of the agent."""
    # Create agent with mock LLM
    agent = RegulationAgent(verbose=True)

    # Register some sample tools for demo
    from .tool_registry import Tool

    def search_regulations(query: str, limit: int = 5) -> List[Dict]:
        """Search regulatory documents."""
        return [
            {"title": f"Documento sobre {query}", "relevance": 0.95},
            {"title": f"Regulación de {query}", "relevance": 0.85},
        ]

    tool = Tool(
        name="search_regulations",
        description="Busca documentos regulatorios relevantes",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Término de búsqueda"},
                "limit": {"type": "integer", "description": "Número máximo de resultados"},
            },
            "required": ["query"],
        },
        function=search_regulations,
        category="search",
    )

    agent.registry.register(tool)

    # Run demo query
    print("=" * 60)
    print("RegLLM Agent Demo")
    print("=" * 60)

    result = agent.run("¿Qué es la metodología IRB?")

    print(f"\nQuery: {result['query']}")
    print(f"State: {result['state']}")
    print(f"Steps: {result['total_steps']}")
    print(f"\nAnswer: {result['answer']}")


if __name__ == "__main__":
    main()
