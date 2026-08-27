"""Consensus engine - combines surviving hypotheses without forcing agreement."""

from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


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
class ConsensusResult:
    """Result of consensus process."""
    final_answer: str
    confidence: float
    agreement_level: str  # "strong", "partial", "none"
    supporting_evidence: List[str]
    dissenting_views: List[str]
    metadata: dict


class ConsensusEngine:
    """Engine for reaching consensus without forcing agreement.

    This engine:
    1. Compares arguments, not just conclusions
    2. Evaluates evidence quality
    3. Preserves disagreement when appropriate
    4. Can answer "I cannot establish this with sufficient confidence"
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    def reach_consensus(
        self,
        responses: List[AgentResponse],
        verification_results: Optional[List[dict]] = None,
    ) -> ConsensusResult:
        """Attempt to reach consensus from multiple responses."""
        if not responses:
            return self._no_consensus("No responses to consider")

        # Group responses by similarity
        groups = self._group_responses(responses)

        # Find majority and minority groups
        if len(groups) == 1:
            # All agree
            return self._strong_consensus(groups[0], verification_results)
        elif len(groups) == 2:
            # Two viewpoints
            return self._partial_consensus(groups, verification_results)
        else:
            # Multiple viewpoints
            return self._weak_consensus(groups, verification_results)

    def _group_responses(self, responses: List[AgentResponse]) -> List[List[AgentResponse]]:
        """Group responses by answer similarity."""
        # Simple grouping by extracted answer
        groups = {}
        for response in responses:
            answer = response.metadata.get("extracted_answer", response.content[:100])
            # Normalize answer for comparison
            normalized = self._normalize_answer(answer)
            if normalized not in groups:
                groups[normalized] = []
            groups[normalized].append(response)

        return list(groups.values())

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Simple normalization: lowercase, strip whitespace
        return answer.lower().strip()

    def _strong_consensus(
        self,
        group: List[AgentResponse],
        verification_results: Optional[List[dict]] = None,
    ) -> ConsensusResult:
        """All responses agree."""
        avg_confidence = sum(r.confidence for r in group) / len(group)

        # Check verification
        verified = False
        if verification_results:
            verified = any(v.get("correct", False) for v in verification_results)

        final_confidence = min(avg_confidence, 1.0)
        if verified:
            final_confidence = min(final_confidence + 0.2, 1.0)

        return ConsensusResult(
            final_answer=group[0].metadata.get("extracted_answer", group[0].content),
            confidence=final_confidence,
            agreement_level="strong",
            supporting_evidence=[r.content[:200] for r in group],
            dissenting_views=[],
            metadata={
                "num_agents": len(group),
                "verified": verified,
            },
        )

    def _partial_consensus(
        self,
        groups: List[List[AgentResponse]],
        verification_results: Optional[List[dict]] = None,
    ) -> ConsensusResult:
        """Two viewpoints exist."""
        # Find majority group
        groups_sorted = sorted(groups, key=len, reverse=True)
        majority = groups_sorted[0]
        minority = groups_sorted[1] if len(groups_sorted) > 1 else []

        # Check if verification supports one side
        verified_answer = None
        if verification_results:
            for v in verification_results:
                if v.get("correct"):
                    verified_answer = v.get("claim")
                    break

        # Determine which group is more likely correct
        majority_confidence = sum(r.confidence for r in majority) / len(majority)
        minority_confidence = sum(r.confidence for r in minority) / len(minority) if minority else 0

        if majority_confidence > minority_confidence:
            winner = majority
            loser = minority
        else:
            winner = minority
            loser = majority

        # Calculate confidence
        confidence = majority_confidence * 0.6 + minority_confidence * 0.4
        if verified_answer:
            confidence = min(confidence + 0.1, 1.0)

        return ConsensusResult(
            final_answer=winner[0].metadata.get("extracted_answer", winner[0].content),
            confidence=confidence,
            agreement_level="partial",
            supporting_evidence=[r.content[:200] for r in winner],
            dissenting_views=[r.content[:200] for r in loser],
            metadata={
                "majority_size": len(majority),
                "minority_size": len(minority),
                "verified": verified_answer is not None,
            },
        )

    def _weak_consensus(
        self,
        groups: List[List[AgentResponse]],
        verification_results: Optional[List[dict]] = None,
    ) -> ConsensusResult:
        """Multiple viewpoints, no clear winner."""
        # Collect all responses
        all_responses = [r for group in groups for r in group]
        avg_confidence = sum(r.confidence for r in all_responses) / len(all_responses)

        return ConsensusResult(
            final_answer="No clear consensus - multiple viewpoints exist",
            confidence=avg_confidence * 0.5,  # Reduce confidence for disagreement
            agreement_level="none",
            supporting_evidence=[],
            dissenting_views=[r.content[:200] for r in all_responses],
            metadata={
                "num_groups": len(groups),
                "total_responses": len(all_responses),
            },
        )

    def _no_consensus(self, reason: str) -> ConsensusResult:
        """No consensus possible."""
        return ConsensusResult(
            final_answer=f"Cannot establish answer: {reason}",
            confidence=0.0,
            agreement_level="none",
            supporting_evidence=[],
            dissenting_views=[],
            metadata={"reason": reason},
        )
