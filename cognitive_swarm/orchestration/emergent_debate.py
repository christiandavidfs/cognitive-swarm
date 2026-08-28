"""Iterative cross-critique debate that tests whether a swarm can EXCEED the oracle.

The single-pass consensus (see optimized_debate.py) can only recombine the two
models' *first guesses* — which mathematically can never beat the oracle (the
best of the two members). To truly test "emergence above the oracle" we need
the debate to produce candidate answers that NO single model produces alone.

We do that with a cross-critique protocol that SEPARATES answer capture from
reasoning capture (the small models answer first, then explain — so we use a
tight "answer-first" prompt for clean answers and a separate reasoning pass
only to feed the critics):

  Stage A — Answer. Each model independently answers with the tight terse prompt
            (the format that already reaches the oracle on easy problems), so we
            get clean extracted answers for the oracle comparison.

  Stage A2 — Reasoning. Each model is asked to reason aloud (larger budget). We
            discard this as the answer but keep the trace to feed the critics.

  Stage B — Cross-critique. Each model is shown the OTHER model's reasoning and
            asked to find the specific error and produce a corrected final
            answer. This is a genuinely different input than the first-pass
            prompt, so it can yield NOVEL candidates (answers neither single
            model produces alone).

  Stage C — Consensus. All candidates are pooled and aggregated with the
            confidence-weighted engine.

Because the crux of emergence is the SELECTION problem, we report two numbers:
  * candidate accuracy — is the correct answer present ANYWHERE in the pool
    (does the debate GENERATE novel knowledge)?
  * system accuracy — the single final answer the consensus returns (can we
    SELECT it?). Majority agreement tends to discard a confident but minority
    correct answer, which is exactly the failure mode that caps us at the oracle.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognitive_swarm.models.multi_backend import MultiModelBackend
from cognitive_swarm.orchestration.prompt_optimizer import PromptOptimizer, TaskType
from cognitive_swarm.orchestration.confidence_consensus import Vote
from cognitive_swarm.orchestration.reasoned_selector import ReasonedSelector

logger = logging.getLogger(__name__)


@dataclass
class EmergentDebateResult:
    problem: str
    phi_answer: str
    qwen_answer: str
    final_answer: str
    confidence: float
    candidates: Dict[str, List[str]] = field(default_factory=dict)
    consensus_metadata: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0


class EmergentDebate:
    """Two-model iterative cross-critique debate."""

    def __init__(self, answer_samples: int = 1, critique_samples: int = 1):
        self.backend = MultiModelBackend()
        self.optimizer = PromptOptimizer(use_llm=False)
        self.consensus = ReasonedSelector()
        self.answer_samples = answer_samples
        self.critique_samples = critique_samples
        self.reason_budget = 96  # for reasoning traces fed to critics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, problem: str) -> EmergentDebateResult:
        start = time.time()
        opt = self.optimizer.optimize(problem, model_name="phi")
        task_type = opt.task_type
        qwen_opt = self.optimizer.optimize(problem, model_name="qwen")

        candidates: Dict[str, List[str]] = {}
        votes: List[Vote] = []
        idx = 0

        def add(model, ans):
            nonlocal idx
            candidates.setdefault(model, []).append(ans)
            votes.append(Vote(model=model, answer=ans, sample_index=idx,
                              is_numeric=self.consensus._is_numeric(ans)))
            idx += 1

        # ---- Stage A: clean answers with the tight prompt (oracle comparison)
        phi_answers = self._answer("phi", opt.optimized_prompt, task_type)
        qwen_answers = self._answer("qwen", qwen_opt.optimized_prompt, task_type)
        for a in phi_answers:
            add("phi", a)
        for a in qwen_answers:
            add("qwen", a)

        # ---- Stage A2: reasoning traces (kept only to feed critics)
        phi_trace = self._reason("phi", problem) or "no reasoning provided."
        qwen_trace = self._reason("qwen", problem) or "no reasoning provided."

        # ---- Stage B: cross-critique produces NOVEL candidates
        qwen_crit = self._critique("qwen", problem, phi_trace, task_type)
        phi_crit = self._critique("phi", problem, qwen_trace, task_type)
        for a in qwen_crit:
            add("qwen_critique", a)
        for a in phi_crit:
            add("phi_critique", a)

        # ---- Stage C: consensus
        verdict = self.consensus.reach_consensus(
            votes, task_type=task_type,
            samples_per_model=max(self.answer_samples, self.critique_samples),
        )

        # Also run the plain v3 engine so we can measure the reasoned
        # selector's NET contribution (did it change the vote-count winner?).
        from cognitive_swarm.orchestration.confidence_consensus import (
            ConfidenceConsensusEngine,
        )
        base = ConfidenceConsensusEngine().reach_consensus(
            votes, task_type=task_type,
            samples_per_model=max(self.answer_samples, self.critique_samples),
        )

        phi_ans = candidates["phi"][0] if candidates.get("phi") else ""
        qwen_ans = candidates["qwen"][0] if candidates.get("qwen") else ""

        return EmergentDebateResult(
            problem=problem,
            phi_answer=phi_ans,
            qwen_answer=qwen_ans,
            final_answer=verdict.final_answer,
            confidence=verdict.confidence,
            candidates=candidates,
            consensus_metadata={
                "selector": verdict.metadata.get("selector", "vote"),
                "winning_cluster": verdict.winning_cluster,
                "cluster_scores": verdict.cluster_scores,
                "num_votes": len(votes),
                "base_answer": base.final_answer,
                "base_selector_changed": base.final_answer != verdict.final_answer,
                "corrections": verdict.metadata.get("corrections", []),
            },
            total_latency_ms=(time.time() - start) * 1000,
        )

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    def _tight_budget(self, task_type: TaskType) -> int:
        return 24 if task_type == TaskType.CODE else 10

    def _answer(self, model, prompt, task_type):
        """Tight-budget generation -> short, clean answer(s)."""
        budget = self._tight_budget(task_type)
        self.backend.load_model(model, max_tokens=budget)
        answers = []
        try:
            for _ in range(self.answer_samples):
                res = self.backend.generate(prompt)
                answers.append(self.extract_terse(res.text))
        finally:
            self.backend.unload_model()
        return answers

    def _reason(self, model, problem):
        """Ask the model to reason aloud (trace kept for critics)."""
        prompt = (
            "Problem: {}\n\n"
            "Solve this step by step. Write out your full reasoning, then end\n"
            "with: FINAL ANSWER: <answer>".format(problem)
        )
        self.backend.load_model(model, max_tokens=self.reason_budget)
        try:
            res = self.backend.generate(prompt)
            return self._reason_text(res.text)
        finally:
            self.backend.unload_model()

    def _critique(self, model, problem, other_trace, task_type):
        label = {"math": "numeric answer", "code": "exact program output",
                 "reasoning": "short logical answer"}.get(task_type.value, "answer")
        prompt = (
            "You are reviewing another AI's solution. Find any mistake and give\n"
            "the CORRECT final answer.\n\n"
            f"Problem: {problem}\n\n"
            "Another AI reasoned:\n---\n"
            f"{other_trace}\n---\n\n"
            "1) Identify any error in that reasoning.\n"
            "2) Recompute the answer yourself carefully.\n"
            "3) End with: FINAL ANSWER: <your {label}>\n\n"
            "FINAL ANSWER: "
        )
        budget = self.reason_budget
        self.backend.load_model(model, max_tokens=budget)
        answers = []
        try:
            for _ in range(self.critique_samples):
                res = self.backend.generate(prompt)
                answers.append(self._extract_marked(res.text))
        finally:
            self.backend.unload_model()
        return answers

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract_terse(self, raw: str) -> str:
        """Clean answer from a tight-budget, terse-prompt generation."""
        if not raw:
            return ""
        s = raw.strip()
        # answer-first: the first standalone number / bool / literal wins.
        m = re.search(
            r"(-?\d+(?:\.\d+)?|\[[^\]]*\]|\{[^}]*\}|\([^)]*\)|true|false|no|yes)",
            s, re.IGNORECASE,
        )
        if m:
            return self._clean(m.group(1))
        lines = [l.strip() for l in s.split("\n") if l.strip()]
        return self._clean(lines[-1]) if lines else self._clean(s)

    def _extract_marked(self, raw: str) -> str:
        """Clean answer from a verbose critique that ends with FINAL ANSWER:."""
        if not raw:
            return ""
        s = raw.strip()
        for marker in ["FINAL ANSWER:", "FINAL ANSWER =", "ANSWER:"]:
            if marker in s:
                tail = s.split(marker, 1)[1]
                for line in tail.split("\n"):
                    line = line.strip().strip("`")
                    if line:
                        return self._clean(line)
                return ""
        # No marker -> answer-first fallback.
        return self.extract_terse(raw)

    def _reason_text(self, raw: str) -> str:
        if not raw:
            return ""
        s = raw.strip()
        m = re.split(r"(?i)\bfinal\s*answer\s*:", s)
        return (m[0].strip() if len(m) > 1 else s)[:600]

    def _clean(self, s: str) -> str:
        s = s.strip().strip("`").strip()
        s = re.sub(r"^(the answer is|answer[:=]|output[:=]|result[:=])\s*", "", s, flags=re.I)
        return s.strip()
