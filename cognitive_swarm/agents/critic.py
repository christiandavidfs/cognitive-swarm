"""Critic agent - attempts to falsify proposed solutions."""

from .base import BaseAgent, AgentContext, AgentResponse


class CriticAgent(BaseAgent):
    """Critic that actively searches for errors in solutions.

    The critic's job is NOT to propose solutions but to find problems:
    - Logical errors
    - Unsupported assumptions
    - Contradictions
    - Missing information
    - Incorrect calculations
    - Hallucinations
    """

    def build_prompt(self, context: AgentContext, **kwargs) -> str:
        """Build critic prompt."""
        solutions = self._format_solutions(context)

        return f"""You are a rigorous critic and falsification expert. Your job is to find ERRORS, not to solve problems.

Problem: {context.task}

Solutions to critique:
{solutions}

Your task:
1. Examine each solution for logical errors
2. Check for unsupported assumptions
3. Identify contradictions between solutions
4. Look for missing information or steps
5. Verify any calculations mentioned
6. Check for hallucinated facts or sources

For each solution, provide:
- Specific errors found (with line/step references)
- Severity: CRITICAL / MODERATE / MINOR
- Explanation of why it's wrong
- Whether the error affects the final answer

If a solution appears correct, say so explicitly but explain why you couldn't falsify it.

Format your response as:
SOLUTION 1 CRITIQUE:
[errors and analysis]

SOLUTION 2 CRITIQUE:
[errors and analysis]

OVERALL ASSESSMENT:
Which solution is more reliable and why."""

    def parse_response(self, response: str, context: AgentContext) -> AgentResponse:
        """Parse critic response."""
        # Count errors found
        critical_count = response.lower().count("critical")
        moderate_count = response.lower().count("moderate")
        minor_count = response.lower().count("minor")

        # Determine if solutions survived criticism
        passed = critical_count == 0

        return AgentResponse(
            agent_name=self.name,
            content=response,
            confidence=0.8 if passed else 0.3,
            reasoning=response,
            metadata={
                "critical_errors": critical_count,
                "moderate_errors": moderate_count,
                "minor_errors": minor_count,
                "solutions_passed": passed,
            },
        )

    def _format_solutions(self, context: AgentContext) -> str:
        """Format solutions from context history for critique."""
        if not context.history:
            return "No solutions provided yet."

        parts = []
        for i, entry in enumerate(context.history):
            agent = entry.get("agent", "unknown")
            content = entry.get("content", "")
            if "solver" in agent.lower():
                parts.append(f"SOLUTION {len(parts) + 1} (from {agent}):\n{content}")

        return "\n\n".join(parts) if parts else "No solutions found to critique."
