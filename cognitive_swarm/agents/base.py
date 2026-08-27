"""Base agent class for all cognitive swarm agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from model inference."""
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    prompt_tokens: int = 0
    total_tokens: int = 0


@dataclass
class AgentContext:
    """Context passed between agents."""
    task: str
    history: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    round_number: int = 0
    max_rounds: int = 3


@dataclass
class AgentResponse:
    """Standard response from any agent."""
    agent_name: str
    content: str
    confidence: float = 0.0
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)
    inference_result: Optional[InferenceResult] = None


class BaseAgent(ABC):
    """Abstract base class for all agents in the cognitive swarm."""

    def __init__(self, name: str, backend):
        self.name = name
        self.backend = backend

    @abstractmethod
    def build_prompt(self, context: AgentContext, **kwargs) -> str:
        """Build the prompt for this agent."""
        pass

    @abstractmethod
    def parse_response(self, response: str, context: AgentContext) -> AgentResponse:
        """Parse the raw model response into structured format."""
        pass

    def run(self, context: AgentContext, **kwargs) -> AgentResponse:
        """Execute the agent's task."""
        logger.info(f"[{self.name}] Running on task: {context.task[:50]}...")

        prompt = self.build_prompt(context, **kwargs)
        inference_result = self.backend.generate(prompt, **kwargs)
        response = self.parse_response(inference_result.text, context)
        response.inference_result = inference_result

        logger.info(
            f"[{self.name}] Completed in {inference_result.latency_ms:.0f}ms, "
            f"{inference_result.tokens_generated} tokens"
        )

        return response

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from response text."""
        import re
        patterns = [
            r"confidence[:\s]*(\d+(?:\.\d+)?)\s*%",
            r"(\d+(?:\.\d+)?)\s*%\s*confident",
            r"certainty[:\s]*(\d+(?:\.\d+)?)\s*%",
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return float(match.group(1)) / 100.0
        return 0.5  # Default confidence
