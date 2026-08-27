"""Debate protocol - manages structured interaction between agents."""

from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


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
    inference_result: object = None


@dataclass
class DebateRound:
    """A single round in the debate."""
    round_number: int
    responses: List[AgentResponse] = field(default_factory=list)
    critiques: List[AgentResponse] = field(default_factory=list)
    verifications: List[AgentResponse] = field(default_factory=list)


@dataclass
class DebateResult:
    """Final result of a debate."""
    rounds: List[DebateRound]
    final_responses: List[AgentResponse]
    consensus: Optional[AgentResponse]
    agreement_reached: bool
    total_rounds: int
    metadata: dict = field(default_factory=dict)


class DebateProtocol:
    """Manages the structured debate between agents.

    Protocol:
    Round 1: Solvers independently solve
    Round 2: Critiques of each other's solutions
    Round 3: Revisions based on critiques
    Round 4: Verification of claims
    Round 5: Consensus attempt

    Stops early if:
    - Agreement reached
    - One solution clearly superior
    - Max rounds exceeded
    """

    def __init__(self, max_rounds: int = 3, early_stop: bool = True):
        self.max_rounds = max_rounds
        self.early_stop = early_stop
        self.rounds: List[DebateRound] = []

    def run_debate(
        self,
        task: str,
        solvers: list,
        critic,
        verifier=None,
        initial_context: Optional[AgentContext] = None,
    ) -> DebateResult:
        """Run a complete debate session."""
        logger.info(f"Starting debate on: {task[:50]}...")

        context = initial_context or AgentContext(task=task)

        # Round 1: Independent solutions
        round1 = self._run_solvers(task, solvers, context)
        self.rounds.append(round1)

        # Update context with solutions
        for response in round1.responses:
            context.history.append({
                "agent": response.agent_name,
                "content": response.content,
                "round": 1,
            })

        # Rounds 2+: Critique and revision
        for round_num in range(2, self.max_rounds + 1):
            logger.info(f"Round {round_num}")

            # Run critic
            critique_round = self._run_critic(task, critic, context)
            self.rounds.append(critique_round)

            # Update context with critiques
            for critique in critique_round.critiques:
                context.history.append({
                    "agent": critique.agent_name,
                    "content": critique.content,
                    "round": round_num,
                    "type": "critique",
                })

            # Check for early stop
            if self.early_stop and self._check_agreement():
                logger.info("Agreement reached, stopping early")
                break

            # Run solvers again with critique feedback
            if round_num < self.max_rounds:
                revision_round = self._run_solvers(task, solvers, context)
                self.rounds.append(revision_round)

                for response in revision_round.responses:
                    context.history.append({
                        "agent": response.agent_name,
                        "content": response.content,
                        "round": round_num + 1,
                        "type": "revision",
                    })

        # Verification round
        if verifier:
            verification_round = self._run_verifier(task, verifier, context)
            self.rounds.append(verification_round)

        # Compile results
        final_responses = self._get_final_responses()
        agreement = self._check_agreement()

        return DebateResult(
            rounds=self.rounds,
            final_responses=final_responses,
            consensus=None,
            agreement_reached=agreement,
            total_rounds=len(self.rounds),
            metadata={
                "task": task,
                "num_solvers": len(solvers),
                "early_stopped": len(self.rounds) < self.max_rounds,
            },
        )

    def _run_solvers(self, task: str, solvers: list, context: AgentContext) -> DebateRound:
        """Run all solvers independently."""
        round_obj = DebateRound(round_number=len(self.rounds) + 1)

        for solver in solvers:
            response = solver.run(context)
            round_obj.responses.append(response)

        return round_obj

    def _run_critic(self, task: str, critic, context: AgentContext) -> DebateRound:
        """Run the critic on current solutions."""
        round_obj = DebateRound(round_number=len(self.rounds) + 1)
        response = critic.run(context)
        round_obj.critiques.append(response)
        return round_obj

    def _run_verifier(self, task: str, verifier, context: AgentContext) -> DebateRound:
        """Run verification on claims."""
        round_obj = DebateRound(round_number=len(self.rounds) + 1)
        response = verifier.run(context)
        round_obj.verifications.append(response)
        return round_obj

    def _check_agreement(self) -> bool:
        """Check if agents agree on the answer."""
        if not self.rounds:
            return False

        last_round = self.rounds[-1]
        if not last_round.responses:
            return False

        # Extract answers from responses
        answers = []
        for response in last_round.responses:
            if hasattr(response, 'metadata') and 'extracted_answer' in response.metadata:
                answers.append(response.metadata['extracted_answer'])

        if len(answers) < 2:
            return False

        # Check if all answers are the same (simplified)
        return len(set(answers)) == 1

    def _get_final_responses(self):
        """Get the final responses from solvers."""
        if not self.rounds:
            return []

        # Get responses from last solver round
        for round_obj in reversed(self.rounds):
            if round_obj.responses:
                return round_obj.responses

        return []
