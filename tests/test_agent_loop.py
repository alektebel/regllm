"""
Tests for Agent Loop
"""

import sys
from pathlib import Path

# Add project root to path to allow direct imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

# Direct imports to avoid src/__init__.py heavy dependencies
from src.agents.tool_registry import Tool, ToolRegistry
from src.agents.agent_loop import (
    RegulationAgent,
    MethodologyConsistencyAgent,
    AgentState,
    AgentMessage,
    AgentStep,
)


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_message_creation(self):
        """Test creating an AgentMessage."""
        msg = AgentMessage(
            role="user",
            content="Hello agent",
        )

        assert msg.role == "user"
        assert msg.content == "Hello agent"
        assert msg.timestamp is not None

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = AgentMessage(
            role="assistant",
            content="Here is my response",
            tool_call={"name": "search", "parameters": {"q": "test"}},
        )

        d = msg.to_dict()

        assert d["role"] == "assistant"
        assert d["content"] == "Here is my response"
        assert d["tool_call"]["name"] == "search"


class TestAgentStep:
    """Tests for AgentStep dataclass."""

    def test_step_creation(self):
        """Test creating an AgentStep."""
        step = AgentStep(
            step_number=1,
            thought="I need to search for information",
            action="search",
            action_input={"query": "IRB"},
        )

        assert step.step_number == 1
        assert step.action == "search"
        assert not step.is_final

    def test_final_step(self):
        """Test creating a final step."""
        step = AgentStep(
            step_number=3,
            thought="I have all the information",
            is_final=True,
            observation="Final answer here",
        )

        assert step.is_final
        assert step.observation == "Final answer here"

    def test_step_to_dict(self):
        """Test converting step to dictionary."""
        step = AgentStep(
            step_number=1,
            thought="Thinking...",
            action="tool_name",
            action_input={"param": "value"},
            observation="Result",
        )

        d = step.to_dict()

        assert d["step"] == 1
        assert d["thought"] == "Thinking..."
        assert d["action"] == "tool_name"


class TestRegulationAgent:
    """Tests for RegulationAgent class."""

    @pytest.fixture
    def agent_with_tools(self):
        """Create agent with sample tools."""
        registry = ToolRegistry()

        def search_docs(query: str) -> list:
            return [{"title": f"Doc about {query}", "relevance": 0.9}]

        def calculate_rwa(pd: float, lgd: float, ead: float) -> dict:
            k = pd * lgd * 12.5
            return {"rwa": k * ead}

        tools = [
            Tool(
                name="search_docs",
                description="Search regulatory documents",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
                function=search_docs,
            ),
            Tool(
                name="calculate_rwa",
                description="Calculate RWA",
                parameters={
                    "type": "object",
                    "properties": {
                        "pd": {"type": "number"},
                        "lgd": {"type": "number"},
                        "ead": {"type": "number"},
                    },
                    "required": ["pd", "lgd", "ead"],
                },
                function=calculate_rwa,
            ),
        ]

        for t in tools:
            registry.register(t)

        return RegulationAgent(registry=registry, max_steps=5)

    def test_agent_initialization(self, agent_with_tools):
        """Test agent initializes correctly."""
        assert agent_with_tools.state == AgentState.IDLE
        assert len(agent_with_tools.conversation) == 0
        assert len(agent_with_tools.steps) == 0

    def test_agent_get_system_prompt(self, agent_with_tools):
        """Test system prompt generation includes tools."""
        prompt = agent_with_tools._get_system_prompt()

        assert "experto" in prompt.lower()
        assert "search_docs" in prompt or "herramienta" in prompt.lower()

    def test_parse_response_with_action(self, agent_with_tools):
        """Test parsing response with tool action."""
        response = """PENSAMIENTO: Necesito buscar información sobre IRB.

ACCIÓN: search_docs
ENTRADA: {"query": "IRB methodology"}"""

        step = agent_with_tools._parse_response(response)

        assert "buscar" in step.thought.lower() or "IRB" in step.thought
        assert step.action == "search_docs"
        assert step.action_input["query"] == "IRB methodology"
        assert not step.is_final

    def test_parse_response_final_answer(self, agent_with_tools):
        """Test parsing response with final answer."""
        response = """PENSAMIENTO: Ya tengo toda la información necesaria.

RESPUESTA_FINAL: El método IRB Fundación requiere estimación propia de PD."""

        step = agent_with_tools._parse_response(response)

        assert step.is_final
        assert "IRB" in step.observation

    def test_execute_step(self, agent_with_tools):
        """Test executing a step with tool action."""
        step = AgentStep(
            step_number=1,
            thought="Search for docs",
            action="search_docs",
            action_input={"query": "capital"},
        )

        observation = agent_with_tools._execute_step(step)

        assert "Doc about capital" in observation or "capital" in observation.lower()

    def test_execute_step_no_action(self, agent_with_tools):
        """Test executing step without action."""
        step = AgentStep(
            step_number=1,
            thought="Just thinking",
        )

        observation = agent_with_tools._execute_step(step)

        assert "No se especificó" in observation

    def test_run_returns_result(self, agent_with_tools):
        """Test that run returns a result dict."""
        result = agent_with_tools.run("¿Qué es IRB?")

        assert "query" in result
        assert "answer" in result
        assert "steps" in result
        assert "state" in result
        assert result["query"] == "¿Qué es IRB?"

    def test_run_with_context(self, agent_with_tools):
        """Test running with additional context."""
        result = agent_with_tools.run(
            "¿Qué parámetros necesito?",
            context="Estamos evaluando IRB Fundación para exposiciones corporativas.",
        )

        assert result["query"] == "¿Qué parámetros necesito?"

    def test_reset_clears_state(self, agent_with_tools):
        """Test that reset clears agent state."""
        # Run a query first
        agent_with_tools.run("Test query")

        # Reset
        agent_with_tools.reset()

        assert agent_with_tools.state == AgentState.IDLE
        assert len(agent_with_tools.conversation) == 0
        assert len(agent_with_tools.steps) == 0

    def test_max_steps_limit(self, agent_with_tools):
        """Test that agent stops at max steps."""
        # Set very low max steps
        agent_with_tools.max_steps = 2

        result = agent_with_tools.run("Complex question")

        assert result["total_steps"] <= 2


class TestMethodologyConsistencyAgent:
    """Tests for MethodologyConsistencyAgent class."""

    @pytest.fixture
    def consistency_agent(self, registered_registry):
        """Create consistency agent with all tools."""
        return MethodologyConsistencyAgent(
            registry=registered_registry,
            max_steps=5,
        )

    def test_consistency_agent_initialization(self, consistency_agent):
        """Test consistency agent initializes correctly."""
        assert consistency_agent.state == AgentState.IDLE
        assert "consistencia" in consistency_agent.SYSTEM_PROMPT.lower()

    def test_check_consistency_returns_result(self, consistency_agent):
        """Test check_consistency returns proper result."""
        result = consistency_agent.check_consistency(
            methodology_path="data/methodology/test.md",
            code_path="src/implementation.py",
            aspects=["PD calculation", "LGD values"],
        )

        assert "query" in result
        assert "answer" in result or result.get("error") is not None

    def test_system_prompt_includes_format(self, consistency_agent):
        """Test system prompt includes report format."""
        prompt = consistency_agent.SYSTEM_PROMPT

        assert "Reporte" in prompt or "reporte" in prompt
        assert "Consistentes" in prompt or "consistentes" in prompt.lower()
        assert "Inconsistencias" in prompt or "inconsistencias" in prompt.lower()
