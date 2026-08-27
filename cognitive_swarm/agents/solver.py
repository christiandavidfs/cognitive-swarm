"""Solver agent - attempts to solve problems independently."""

from .base import BaseAgent, AgentContext, AgentResponse
import re


class SolverAgent(BaseAgent):
    """Independent solver that attempts to solve problems.

    Two solver instances can use different prompts/strategies:
    - Solver A: Normal approach
    - Solver B: Independent verification approach
    """

    def __init__(
        self,
        name: str,
        backend,
        strategy: str = "normal",
    ):
        super().__init__(name, backend)
        self.strategy = strategy

    def build_prompt(self, context: AgentContext, **kwargs) -> str:
        """Build solver prompt based on strategy."""
        if self.strategy == "normal":
            return f"""Solve this problem step by step, then give your final answer.

Problem: {context.task}

Think briefly, then end with:
ANSWER: [your answer]"""

        elif self.strategy == "skeptical":
            return f"""Solve this problem independently. Do NOT assume previous work is correct.

Problem: {context.task}

Verify each step, then end with:
ANSWER: [your answer]"""

        elif self.strategy == "creative":
            return f"""Find an alternative solution to this problem.

Problem: {context.task}

Try a different approach, then end with:
ANSWER: [your answer]"""

        else:
            return f"""Solve: {context.task}

ANSWER: [your answer]"""

    def parse_response(self, response: str, context: AgentContext) -> AgentResponse:
        """Parse solver response."""
        confidence = self._extract_confidence(response)
        answer = self._extract_answer(response)

        return AgentResponse(
            agent_name=self.name,
            content=response,
            confidence=confidence,
            reasoning=response,
            metadata={
                "strategy": self.strategy,
                "extracted_answer": answer,
            },
        )

    def _build_context_string(self, context: AgentContext) -> str:
        """Build context string from history."""
        if not context.history:
            return "No previous reasoning available."

        parts = []
        for entry in context.history[-3:]:  # Last 3 entries
            agent = entry.get("agent", "unknown")
            content = entry.get("content", "")[:500]
            parts.append(f"[{agent}]: {content}")
        return "\n\n".join(parts)

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response - prefer ANSWER: format."""
        # First try to find explicit ANSWER: format
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try other common patterns
        patterns = [
            r"(?:final answer|answer|result)[:\s]*(.+?)(?:\n|$)",
            r"(?:therefore|thus|so)[,\s]+(.+?)(?:\n|$)",
            r"(?:the answer is)[:\s]*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return match.group(1).strip()

        # If no pattern matches, return last line
        lines = response.strip().split("\n")
        return lines[-1] if lines else response
