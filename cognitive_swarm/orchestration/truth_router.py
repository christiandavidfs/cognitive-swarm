"""Verifier / router: the real-life truth hierarchy for the swarm.

Real learning uses the cheapest, most-certain *source of truth* available —
not consensus between guesses. Given a question we route through tiers, and we
ALWAYS return a structured Verdict (answer + confidence + sources +
disagreement) instead of a bare atomic answer, so we can be honest when a
contested question has no single truth:

  Tier 0  MEMORY    : already-verified (provenance-aware). Instant, no compute.
  Tier 1  CODE EXEC : code-output question -> run the interpreter.
                       Deterministic ground truth, ms, no model load.
  Tier 2  DETERM    : deterministic primitives, checked from most-
                       to least-specific:
                         2b  string-op   (reverse/count/compare)
                         2c  math-prims  (factorial, sqrt, log, power,
                                         geometry, multi-step)
                         2  calculator  (literal ``a op b``)
  Tier 3  RETRIEVAL : factual/contested (history, chemistry-opinion) ->
                       collect claims from ALL registered retrievers
                       (local docs, a knowledge table, and any future
                       connector), merge and corroborate
                       reliability-weighted; surface conflict, never
                       fake a single answer for ambiguous domains.
  Tier 4  DEBATE    : weak-model debate as a HYPOTHESIS source; any
                       candidate a ground-truth check can confirm is
                       elevated/archived.

Tier 1/2 results are archived to memory with their provenance. Disagreement is
preserved in memory so the swarm's experience keeps conflicts rather than
erasing them.

The retriever layer is pluggable and media-agnostic: any object implementing
``Retriever.get_claims(question) -> list[SourceClaim]`` (see
``retrievers.py``) can be registered, so a live document store, a database, or
a web API can be swapped in without touching the router.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from cognitive_swarm.tools.executor import execute as _exec_code
from cognitive_swarm.tools.string_primitives import resolve_string_primitive as _resolve_string_primitive
from cognitive_swarm.tools.math_primitives import (
    resolve_math_primitive as _resolve_math_primitive,
    resolve_compound_expression as _resolve_compound_expression,
)
from cognitive_swarm.memory.verified_memory import VerifiedMemory
from cognitive_swarm.orchestration.prompt_optimizer import PromptOptimizer, TaskType
from cognitive_swarm.orchestration.corroboration import Corroborator, SourceClaim, Verdict
from cognitive_swarm.orchestration.retrievers import Retriever


@dataclass
class Resolution:
    """Structured answer: never a bare atom; always carries provenance."""
    answer: Optional[str]
    confidence: float
    status: str                    # memory|executor|calculator|retrieval|debate|none
    tier: str
    sources: List[dict] = field(default_factory=list)
    disagreement: List[dict] = field(default_factory=list)


class TruthRouter:
    """Routes a question to the strongest available source of truth."""

    def __init__(self, memory: Optional[VerifiedMemory] = None,
                 corroborator: Optional[Corroborator] = None,
                 retrievers: Optional[List[Retriever]] = None):
        self.memory = memory or VerifiedMemory()
        self.corroborator = corroborator or Corroborator()
        self.retrievers: List[Retriever] = list(retrievers or [])
        self.optimizer = PromptOptimizer(use_llm=False)

    def add_retriever(self, retriever: Retriever) -> None:
        """Register a connector; the router queries all of them at Tier 3."""
        self.retrievers.append(retriever)

    # ------------------------------------------------------------------
    # Main entry: resolve a question with the cheapest certain source.
    # ------------------------------------------------------------------
    def resolve(self, question: str) -> Resolution:
        # Tier 0: memory first (experience beats recomputation).
        rec = self.memory.lookup(question)
        if rec is not None:
            return Resolution(
                answer=rec["answer"], confidence=rec.get("confidence", 1.0),
                status="memory", tier=rec.get("tier", "memory"),
                sources=rec.get("sources", []),
                disagreement=rec.get("disagreement", []),
            )

        task_type = self.optimizer.optimize(question).task_type

        # Tier 1: code -> run it (ground truth).
        if task_type == TaskType.CODE:
            out = _exec_code(question)
            if out is not None:
                self._verify(question, out, "executor", tier="executor")
                return Resolution(out, 1.0, "executor", "executor",
                                  sources=[{"name": "python-interpreter", "reliability": 1.0}])

        # Tier 2b: deterministic string-op primitives (reverse, count char/
        # vowels/letters, fraction-vs-decimal compare). These are more specific
        # than generic arithmetic, so they are checked BEFORE the calculator:
        # e.g. "which is larger: 3/10 or 1/3" is a compare, not a division to
        # evaluate. Cached ``thinking primitives``: resolve the classic
        # adversarial trap items exactly at zero compute instead of re-deriving
        # (or asking a model) each time.
        prim = _resolve_string_primitive(question)
        if prim is not None:
            self._verify(question, prim, "string-op", tier="string-op")
            return Resolution(prim, 1.0, "string-op", "string-op",
                              sources=[{"name": "string-primitives", "reliability": 1.0}])

        # Tier 2c: math primitives for word operators / special forms.
        # Compound expressions (multi-step arithmetic) are tried first
        # to capture cases like ``100 / 4 + 6`` with proper precedence.
        # Specific primitives (factorial, sqrt, log, series sum, etc.) are
        # tried second to handle unambiguous cases.
        # Cached thinking primitives -> zero model load, ms.
        if task_type == TaskType.MATH:
            mp = _resolve_compound_expression(question)
            if mp is None:
                mp = _resolve_math_primitive(question)
            if mp is not None:
                self._verify(question, mp, "math-primitives", tier="math-primitives")
                return Resolution(mp, 1.0, "math-primitives", "math-primitives",
                                  sources=[{"name": "math-primitives", "reliability": 1.0}])

        # Tier 2: self-contained arithmetic.
        if task_type == TaskType.MATH:
            calc = self._try_calc(question)
            if calc is not None:
                self._verify(question, calc, "calculator", tier="calculator")
                return Resolution(calc, 1.0, "calculator", "calculator",
                                  sources=[{"name": "calculator", "reliability": 1.0}])

        # Tier 3: factual / contested -> corroborate available sources.
        claims = self._collect_source_claims(question, task_type)
        if claims:
            verdict = self.corroborator.corroborate(claims)
            if verdict.status == "corroborated":
                self._verify(question, verdict.answer, "retrieval", tier="retrieval",
                             confidence=verdict.confidence,
                             reliability=verdict.reliability,
                             sources=verdict.provenance,
                             disagreement=verdict.disagreement)
                return Resolution(verdict.answer, verdict.confidence, "retrieval",
                                  "retrieval", sources=verdict.provenance,
                                  disagreement=verdict.disagreement)
            # Conflict or uncertain -> surface honestly, store the conflict.
            sources = [{"name": cl["sources"], "reliability": cl["best_reliability"]}
                       for cl in verdict.clusters.values()]
            return Resolution(None, verdict.confidence, "retrieval", "conflict",
                              sources=sources, disagreement=verdict.disagreement)

        # Tier 4: nothing deterministic -> None (caller falls back to debate).
        return Resolution(None, 0.0, "none", "debate")

    # ------------------------------------------------------------------
    # Verification / archival helpers
    # ------------------------------------------------------------------
    def verify_candidate(self, question: str, candidate: str,
                         source: str = "debate") -> bool:
        """Confirm a pooled candidate against any ground-truth check."""
        if self.optimizer.optimize(question).task_type == TaskType.CODE:
            out = _exec_code(question)
            if out is not None and self._answers_equal(out, candidate):
                self._verify(question, out, "executor", tier="executor")
                return True
            return False
        calc = self._try_calc(question)
        if calc is not None:
            if self._answers_equal(calc, candidate):
                self._verify(question, calc, "calculator", tier="calculator")
                return True
            return False
        return False

    def _verify(self, question, answer, source, tier, confidence=1.0,
                reliability=1.0, sources=None, disagreement=None) -> None:
        self.memory.remember(
            question, answer, source=source, tier=tier,
            confidence=confidence, reliability=reliability,
            sources=sources, disagreement=disagreement,
        )

    # ------------------------------------------------------------------
    # Retrieval / source collection (Tier 3)
    # ------------------------------------------------------------------
    def _collect_source_claims(self, question: str,
                               task_type: Optional[TaskType] = None) -> List[SourceClaim]:
        """Gather candidate facts for a factual/contested question.

        This is the Tier 3 **gate**: only connectors whose declared
        ``categories`` match the question's category are queried. A code or
        arithmetic question skips factual connectors entirely (path of least
        resistance), so we neither waste work nor inject irrelevant verbose
        claims into corroboration. Each selected connector returns its own
        ``SourceClaim`` objects (reliability + independence flag), merged here
        before corroboration.
        """
        claims: List[SourceClaim] = []
        for retriever in self.retrievers:
            if not retriever.applies_to(task_type):
                continue
            try:
                claims.extend(retriever.get_claims(question))
            except Exception:
                # One failing connector must not take down the whole tier.
                continue
        return claims


    # ------------------------------------------------------------------
    # Small deterministic calculators
    # ------------------------------------------------------------------
    def _try_calc(self, question: str) -> Optional[str]:
        """Evaluate a simple, self-contained arithmetic question if safe."""
        q = question.lower()
        m = re.search(
            r"(-?\d+(?:\.\d+)?)\s*([+\-*/x^])\s*(-?\d+(?:\.\d+)?)", q
        )
        if not m:
            return None
        a, op, b = m.group(1), m.group(2), m.group(3)
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return None
        if op == "+":
            val = fa + fb
        elif op == "-":
            val = fa - fb
        elif op in ("*", "x"):
            val = fa * fb
        elif op == "/":
            if fb == 0:
                return None
            val = fa / fb
        else:
            return None
        return self._fmt(val)

    @staticmethod
    def _fmt(x: float) -> str:
        if x == int(x):
            return str(int(x))
        return ("%.6f" % x).rstrip("0").rstrip(".")

    @staticmethod
    def _answers_equal(a: str, b: str) -> bool:
        a, b = a.strip().lower(), b.strip().lower()
        if a == b:
            return True
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (ValueError, TypeError):
            return False


def demo_fact_table(router: TruthRouter) -> None:
    """Register demonstration retrievers to demo Tier 3.

    Registers a hand-authored knowledge table AND a live ``LocalDocsRetriever``
    over the committed ``retrieval_corpus/`` documents. Together they show a
    CORROBORATED case (two independent sources agree) and a CONFLICT case (two
    reliable sources disagree -> honest disagreement report), and exercise the
    real retrieval tier end to end.
    """
    from cognitive_swarm.orchestration.prompt_optimizer import TaskType
    from cognitive_swarm.orchestration.retrievers import LocalDocsRetriever, TableRetriever

    factual = [TaskType.REASONING, TaskType.UNKNOWN]

    table = {
        "fall of the western roman empire": [
            {"name": "primary: contemporary chronicle", "answer": "476 CE",
             "reliability": 0.9, "independent": True,
             "reason": "primary source, contemporary"},
            {"name": "peer-reviewed history textbook", "answer": "476 CE",
             "reliability": 0.9, "independent": True,
             "reason": "independent secondary source"},
            {"name": "popular encyclopaedia", "answer": "476 CE",
             "reliability": 0.7, "independent": False,
             "reason": "derives from the above"},
        ],
        "what caused the fall of the roman empire": [
            {"name": "historian A (economic school)", "answer": "economic decline",
             "reliability": 0.8, "independent": True},
            {"name": "historian B (barbarian invasions school)", "answer": "barbarian invasions",
             "reliability": 0.8, "independent": True},
        ],
    }
    router.add_retriever(TableRetriever(table, categories=factual))
    router.add_retriever(LocalDocsRetriever(
        Path(__file__).parent.parent.parent / "retrieval_corpus", name="local-docs",
        categories=factual,
    ))
