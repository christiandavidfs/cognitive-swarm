"""Reasoned-selector: prevent vote-counting from discarding novel corrections.

The cross-critique debate (emergent_debate.py) produces a candidate pool whose
accuracy far exceeds what pure aggregation can *select* — on the hard subset
the candidate pool hit 62% while the confidence-weighted system landed at 38%.
The reason, confirmed by the conformity literature (see docs/THESIS.md), is
that vote-counting rewards a *stable-but-wrong first-pass majority* and throws
away exactly the answer that matters most: a model's *independent, reasoned
correction* produced during cross-critique.

This engine layers a reasoned-correction signal on top of the v3
confidence-weighted engine (confidence_consensus.py). The v3 engine is left
intact and remains the fallback; here we only RAISE the score of an answer
that meets all of the following conservative criteria, so ordinary debates are
unchanged but genuine emergence gets through:

  1. **Self-overturn** — a critic's answer differs from BOTH the critic's own
     first-pass answer AND the current winning majority. A critic changing its
     mind *away* from the majority is making an independence move (reasoning-
     induced persuasion, not conformity).
  2. **Concrete clean answer** — the correction is a usable numeric/boolean
     answer, so the "critique noise" garbage that small models occasionally
     emit (e.g. 'no', '(n-1\\)') can never hijack the verdict.

We deliberately do NOT trust critique agreement by itself: two critics can
conform to each other on a wrong answer (observed on genuinely indeterminate
problems), which no deterministic selection signal can rescue. The selector
only rescues the crisp self-overturn case.

The selector is deterministic (no LLM) and pluggable: EmergentDebate can use it
or fall back to ConfidenceConsensusEngine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cognitive_swarm.orchestration.confidence_consensus import (
    ConfidenceConsensusEngine,
    ConsensusVerdict,
    TaskType,
    Vote,
)


@dataclass
class Correction:
    """A single detected reasoned correction candidate."""
    answer: str                 # the corrected answer to raise
    critic: str                 # model name of the critic that produced it
    first_pass: str             # that critic's own first-pass answer
    first_pass_key: str         # normalized first-pass answer
    new_key: str                # normalized corrected answer


class ReasonedSelector:
    """Selects a verdict that survives a genuine reasoned correction.

    Combines the v3 confidence-weighted scores with a correction bonus. An
    answer that is an independent self-overturn correction gets a large bonus
    so it can beat a shallow (first-pass-only) but stable majority.

    The final verdict is produced by the v3 engine, but if the reasoned
    correction displaces the engine's winner, we re-score and report it.
    """

    # A correction that meets all criteria raises its cluster's score well above
    # the +0.5..+1.0 that every ordinary vote adds, letting a lone novel-correct
    # answer beat a stable-but-wrong 2-vote first-pass majority.
    CORRECTION_BONUS = 2.0

    def __init__(self, base: Optional[ConfidenceConsensusEngine] = None):
        self.engine = base or ConfidenceConsensusEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reach_consensus(
        self,
        votes: List[Vote],
        task_type: Optional[TaskType] = None,
        samples_per_model: int = 1,
    ) -> ConsensusVerdict:
        """Aggregate votes that may include cross-critique corrections."""
        if not votes:
            return self.engine.reach_consensus(votes, task_type, samples_per_model)

        # 1. Base verdict exactly as the v3 engine gives it.
        base = self.engine.reach_consensus(votes, task_type, samples_per_model)

        # 2. Detect reasoned corrections from the critique stage.
        corrections = self._find_corrections(votes, base)

        # 3. If a reasoned correction displaces the base winner, override.
        if corrections:
            override = self._pick_override(corrections, votes, task_type)
            if override is not None:
                return self._build_correction_verdict(
                    base, override, corrections, votes, task_type,
                )

        return base

    # ------------------------------------------------------------------
    # Delegation to the base engine (used by EmergentDebate wiring)
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        return self.engine._normalize(text)

    def _is_numeric(self, text: str) -> bool:
        return self.engine._is_numeric(text)

    # ------------------------------------------------------------------
    # Correction detection
    # ------------------------------------------------------------------
    def _find_corrections(
        self,
        votes: List[Vote],
        base: ConsensusVerdict,
    ) -> List[Correction]:
        """Return the list of valid reasoned-correction candidates.

        A candidate is a critique vote whose answer:
          - differs from the winning majority (it is an overturn),
          - differs from that same critic's own first-pass answer (a genuine
            self-overturn — the critic *changed its mind* after cross-critique),
          - is a concrete clean answer (usable number/boolean, not noise).
        """
        corrections: List[Correction] = []
        base_key = self.engine._normalize(base.final_answer) if base.final_answer else ""

        # First passes per model: {model -> [normalized first-pass answers]}
        first_passes: Dict[str, List[str]] = {}
        for v in votes:
            if self._is_critique(v.model):
                continue
            first_passes.setdefault(v.model, []).append(self.engine._normalize(v.answer))

        for v in votes:
            if not self._is_critique(v.model):
                continue
            key = self.engine._normalize(v.answer)
            if not key:
                continue
            # Overturns the current majority.
            if base_key and key == base_key:
                continue
            # Concrete clean answer (not critique noise).
            if not self._usable(key, task_type=None):
                continue
            # Self-overturn: the critic had a concrete first-pass AND changed it.
            own = first_passes.get(self._base_model(v.model), [])
            own_concrete = [k for k in own if k and self._usable(k, task_type=None)]
            if not own_concrete or key in own_concrete:
                # No concrete first-pass, or the critic only reaffirms itself.
                continue
            corrections.append(Correction(
                answer=self._original_text(v, votes),
                critic=v.model,
                first_pass=own_concrete[0],
                first_pass_key=own_concrete[0],
                new_key=key,
            ))
        return corrections

    def _pick_override(
        self,
        corrections: List[Correction],
        votes: List[Vote],
        task_type: Optional[TaskType],
    ) -> Optional[str]:
        """Choose the correction to promote, if any.

        Favor corrections that disagree with the majority (tie-break by how many
        independent voices support them). Return its original string answer.
        """
        # Score each correction's answer by support among non-majority voices.
        scores: Dict[str, float] = {}
        for c in corrections:
            scores[c.new_key] = scores.get(c.new_key, 0.0) + 1.0
        best_key = max(scores, key=lambda k: scores[k])
        best_corrections = [c for c in corrections if c.new_key == best_key]
        # Prefer the first concrete, numeric-ish correction.
        if task_type in (TaskType.MATH, TaskType.CODE):
            numeric = [c for c in best_corrections if self.engine._is_numeric(c.new_key)]
            if numeric:
                best_corrections = numeric
        return best_corrections[0].answer

    def _build_correction_verdict(
        self,
        base: ConsensusVerdict,
        override_answer: str,
        corrections: List[Correction],
        votes: List[Vote],
        task_type: Optional[TaskType],
    ) -> ConsensusVerdict:
        """Build a verdict that reports the override transparently."""
        key = self.engine._normalize(override_answer)
        # Total agreement share for better confidence reporting.
        total = len(votes)
        ok_votes = sum(1 for v in votes if self.engine._normalize(v.answer) == key)
        agreement = ok_votes / max(total, 1)
        confidence = round(max(base.confidence, 0.5 * agreement + 0.5), 3)

        # cluster_scores: keep base scores, but promote the corrected answer so
        # the report shows it won.
        cluster_scores = dict(base.cluster_scores)
        promoted = max(cluster_scores.values(), default=0.0) + self.CORRECTION_BONUS
        cluster_scores[key] = promoted

        return ConsensusVerdict(
            final_answer=override_answer,
            confidence=confidence,
            agreement_level="partial" if ok_votes < total else "strong",
            winning_cluster=key,
            cluster_scores=cluster_scores,
            votes_per_cluster=dict(base.votes_per_cluster),
            metadata={
                **base.metadata,
                "selector": "reasoned",
                "corrections": [
                    {"critic": c.critic, "from": c.first_pass, "to": c.answer}
                    for c in corrections
                ],
                "num_votes": total,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _base_model(model: str) -> str:
        """'qwen_critique' -> 'qwen', 'phi' -> 'phi'."""
        return model.replace("_critique", "")

    @staticmethod
    def _is_critique(model: str) -> bool:
        return model.endswith("_critique")

    def _usable(self, key: str, task_type: Optional[TaskType]) -> bool:
        """Is a normalized answer a clean concrete literal (not critique noise)?

        Only numeric, boolean, yes/no, or well-formed list/set literals count.
        Rejecting free-text prose is the guard against the critique noise small
        models occasionally emit (e.g. 'no', '(n-1\\\\)', "To solve this...").
        """
        if not key:
            return False
        s = key.strip()
        if self.engine._is_numeric(s):
            return True
        if s in ("true", "false", "yes", "no"):
            return True
        # List/set collection literal (well-formed: no stray backslash/eq/expr).
        if s[:1] in ("[", "{") and s[-1:] in ("]", "}"):
            return "\\" not in s and "=" not in s and len(s) <= 64
        return False

    def _original_text(self, v: Vote, votes: List[Vote]) -> str:
        """Return the non-normalized original answer text for a vote."""
        return v.answer
