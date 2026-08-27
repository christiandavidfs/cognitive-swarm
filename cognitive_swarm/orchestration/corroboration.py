"""Corroboration: reliability-weighted agreement across multiple sources.

Real-life truth for non-deterministic domains (history, chemistry models,
disputed facts) rarely comes from a single authority. It comes from *source
criticism*: how reliable is each source, how independent are they, and how
much do they agree?

This module turns a set of clashing source claims into an honest verdict:

  * CORROBORATED   : independent, high-reliability sources converge.
  * CONFLICT       : reliable sources disagree -> we DO NOT force a single
                     answer; we surface a disagreement report with provenance.
  * UNCERTAIN      : too little signal -> low confidence, no claim of truth.

Weights combine each source's declared reliability with an independence bonus,
so that 10 sources that all copy one source count far less than 2 independent
primary sources. This mirrors how a historian treats corroboration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SourceClaim:
    source: str
    answer: str
    reliability: float = 0.5      # 0..1 how trustworthy the source is
    independent: bool = True      # False if it derives from another source
    reason: str = ""              # optional justification


@dataclass
class Verdict:
    status: str                    # "corroborated" | "conflict" | "uncertain"
    answer: Optional[str]
    confidence: float
    reliability: float
    clusters: Dict[str, dict] = field(default_factory=dict)   # answer -> stats
    disagreement: List[dict] = field(default_factory=list)
    provenance: List[dict] = field(default_factory=list)


class Corroborator:
    """Aggregate clashing source claims into an honest verdict."""

    INDEPENDENCE_BONUS = 0.5

    def corroborate(self, claims: List[SourceClaim]) -> Verdict:
        if not claims:
            return Verdict("uncertain", None, 0.0, 0.0)

        # Group claims by normalized answer.
        clusters: Dict[str, dict] = {}
        for c in claims:
            key = self._norm(c.answer)
            if key not in clusters:
                clusters[key] = {
                    "answer": c.answer,
                    "weight": 0.0,
                    "best_reliability": 0.0,
                    "sources": [],
                    "independent_count": 0,
                }
            w = c.reliability
            if c.independent:
                w += self.INDEPENDENCE_BONUS
            clusters[key]["weight"] += w
            clusters[key]["best_reliability"] = max(
                clusters[key]["best_reliability"], c.reliability)
            clusters[key]["independent_count"] += 1 if c.independent else 0
            clusters[key]["sources"].append(c.source)

        total_weight = sum(cl["weight"] for cl in clusters.values())
        if total_weight <= 0:
            return Verdict("uncertain", None, 0.0, 0.0)

        # Winner by weight.
        best_key = max(clusters, key=lambda k: clusters[k]["weight"])
        best = clusters[best_key]
        majority_share = best["weight"] / total_weight if total_weight else 0.0

        # Corroborated if the winner has a dominant weight AND at least one
        # independent source.
        corroborated = (
            best["independent_count"] >= 1
            and majority_share >= 0.6
        )

        # Confidence rises with share and best reliability.
        confidence = round(0.5 * majority_share + 0.5 * best["best_reliability"], 3)

        # Build the disagreement report for every losing cluster.
        disagreement = []
        for key, cl in clusters.items():
            if key == best_key:
                continue
            disagreement.append({
                "answer": cl["answer"],
                "sources": cl["sources"],
                "best_reliability": cl["best_reliability"],
                "weight_share": round(cl["weight"] / total_weight, 3),
            })

        provenance = [{
            "answer": cl["answer"],
            "sources": cl["sources"],
            "best_reliability": cl["best_reliability"],
            "weight_share": round(cl["weight"] / total_weight, 3),
        } for cl in clusters.values()]

        if corroborated:
            return Verdict("corroborated", best["answer"], confidence,
                           best["best_reliability"], clusters,
                           disagreement, provenance)
        # Not corroborated: if there ARE conflicting clusters, it's a conflict;
        # otherwise uncertain.
        if len(clusters) > 1:
            return Verdict("conflict", None, confidence,
                           best["best_reliability"], clusters, disagreement,
                           provenance)
        return Verdict("uncertain", best["answer"], confidence,
                       best["best_reliability"], clusters, disagreement,
                       provenance)

    @staticmethod
    def _norm(s: str) -> str:
        return s.strip().lower()
