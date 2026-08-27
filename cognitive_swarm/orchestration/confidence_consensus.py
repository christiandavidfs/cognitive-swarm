"""Confidence-weighted consensus engine.

The standard consensus engine picks an answer with hard-coded heuristics
(e.g. "prefer the numeric answer", "prefer the shorter answer"), which throws
away valuable information and can discard the single correct minority answer.

This engine treats each model's answer as a *vote* carrying a weight, and the
confidence of a vote comes from three reinforced signals:

1. **Agreement** — how many votes support the same answer (majority mass).
2. **Stability / self-consistency** — whether a model gives the *same* answer
   when sampled more than once. A model that flip-flops between samples is
   unreliable and gets down-weighted. This is the well-known self-consistency
   technique, adapted to a two-model swarm.
3. **Reported confidence** — the model's own 0..1 confidence, used only as a
   *tie-breaker* because small models tend to over-report it.

Each unique answer accumulates a weight = sum over votes of (0.5 agreement
share + stability weight). Mathematically we combine them per answer cluster:

    weight(c) = sum_{vote in c} ( 1.0 for agreement ) * stability_factor(vote)

The cluster with the highest total weight wins; ties are broken by the model's
reported confidence, then by task-type-aware preference (math prefers numeric).

This improves on the naive engine by never *silently discarding* a minority
answer that is stable and confident.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TaskType(Enum):
    REASONING = "reasoning"
    MATH = "math"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class Vote:
    """A single model answer sample to be aggregated."""
    model: str
    answer: str
    reported_confidence: float = 0.5
    sample_index: int = 0
    is_numeric: bool = False


@dataclass
class ConsensusVerdict:
    """Final decision produced by the engine."""
    final_answer: str
    confidence: float
    agreement_level: str          # "strong", "partial", "none"
    winning_cluster: str
    cluster_scores: Dict[str, float] = field(default_factory=dict)
    votes_per_cluster: Dict[str, int] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class ConfidenceConsensusEngine:
    """Aggregates multiple model answers into a confidence-weighted verdict.

    The engine is purely deterministic and does not call an LLM, so it is fast
    and reproducible. `samples_per_model` tells it how many samples were
    collected per model, which it uses to compute per-model stability.
    """

    TIE_EPSILON = 1e-9

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reach_consensus(
        self,
        votes: List[Vote],
        task_type: Optional[TaskType] = None,
        samples_per_model: int = 1,
    ) -> ConsensusVerdict:
        """Aggregate a list of votes into a final verdict."""
        if not votes:
            return ConsensusVerdict(
                final_answer="", confidence=0.0, agreement_level="none",
                winning_cluster="", cluster_scores={},
            )

        # 1. Normalize answers so we can cluster equivalent ones.
        clusters: Dict[str, Dict[str, object]] = {}
        for v in votes:
            key = self._normalize(v.answer)
            if key not in clusters:
                clusters[key] = {
                    "key": key,
                    "votes": [],
                    "models": set(),
                    "count": 0,
                    "reported_conf_sum": 0.0,
                }
            clusters[key]["votes"].append(v)
            clusters[key]["count"] += 1
            clusters[key]["models"].add(v.model)
            clusters[key]["reported_conf_sum"] += v.reported_confidence

        # 2. Compute per-model stability: fraction of that model's samples
        #    that landed in each cluster.
        model_samples: Dict[str, int] = {v.model: 0 for v in votes}
        for v in votes:
            model_samples[v.model] += 1
        # sample count per model could be less than samples_per_model if a
        # generation failed; use actual counts.

        cluster_scores: Dict[str, float] = {}
        votes_per_cluster: Dict[str, int] = {}
        for key, c in clusters.items():
            score = 0.0
            for v in c["votes"]:
                # Fraction of this model's samples in THIS cluster -> stability.
                total_for_model = max(model_samples.get(v.model, 1), 1)
                stability = self._stability_for_model(v.model, key, clusters, model_samples)
                score += 0.5 + 0.5 * stability
            cluster_scores[key] = score
            votes_per_cluster[key] = c["count"]

        # 3. Pick the winning cluster by score.
        best_key = max(cluster_scores, key=lambda k: cluster_scores[k])

        # 4. Resolve near-ties with task-type heuristics and reported confidence.
        winners = [k for k in cluster_scores
                   if abs(cluster_scores[k] - cluster_scores[best_key]) < self.TIE_EPSILON]
        if len(winners) > 1:
            best_key = self._tie_break(winners, clusters, task_type)

        # 5. Decorate result.
        answer = clusters[best_key]["votes"][0].answer
        confidence = self._agreement_level(best_key, clusters, votes)
        agreement = self._level(best_key, clusters, votes)

        return ConsensusVerdict(
            final_answer=answer,
            confidence=confidence,
            agreement_level=agreement,
            winning_cluster=best_key,
            cluster_scores=cluster_scores,
            votes_per_cluster=votes_per_cluster,
            metadata={
                "num_votes": len(votes),
                "num_clusters": len(clusters),
                "winning_votes": votes_per_cluster[best_key],
                "winning_models": sorted(clusters[best_key]["models"]),
            },
        )

    # ------------------------------------------------------------------
    # Stability helpers
    # ------------------------------------------------------------------
    def _stability_for_model(
        self,
        model: str,
        key: str,
        clusters: Dict[str, Dict[str, object]],
        model_samples: Dict[str, int],
    ) -> float:
        """Fraction of a model's samples that agree with this cluster."""
        total = max(model_samples.get(model, 1), 1)
        agree = 0
        for k, c in clusters.items():
            for v in c["votes"]:
                if v.model == model:
                    if k == key:
                        agree += 1
        return agree / total

    def _tie_break(
        self,
        winners: List[str],
        clusters: Dict[str, Dict[str, object]],
        task_type: Optional[TaskType],
    ) -> str:
        """Break a near-tie using reported confidence then task heuristics."""
        # a) Highest aggregate reported confidence.
        best = winners[0]
        best_conf = clusters[best]["reported_conf_sum"] / max(clusters[best]["count"], 1)
        for k in winners[1:]:
            conf = clusters[k]["reported_conf_sum"] / max(clusters[k]["count"], 1)
            if conf > best_conf + self.TIE_EPSILON:
                best, best_conf = k, conf

        # b) Task-type preference.
        if task_type == TaskType.MATH:
            for k in winners:
                if self._is_numeric(clusters[k]["key"]):
                    return k
        return best

    def _agreement_level(
        self,
        best_key: str,
        clusters: Dict[str, Dict[str, object]],
        votes: List[Vote],
    ) -> float:
        """Estimate 0..1 confidence for the chosen answer."""
        total = len(votes)
        best_count = clusters[best_key]["count"]
        agreement = best_count / max(total, 1)
        # Blend in majority share and average stability across the cluster.
        stabilities = []
        model_samples = {v.model: 0 for v in votes}
        for v in votes:
            model_samples[v.model] += 1
        for v in clusters[best_key]["votes"]:
            stabilities.append(self._stability_for_model(v.model, best_key, clusters, model_samples))
        avg_stability = sum(stabilities) / max(len(stabilities), 1)
        return round(0.6 * agreement + 0.4 * avg_stability, 3)

    def _level(self, best_key: str, clusters: Dict[str, Dict[str, object]], votes: List[Vote]) -> str:
        """Classify agreement as strong / partial / none."""
        count = clusters[best_key]["count"]
        total = len(votes)
        if total >= 2 and count == total:
            return "strong"
        if total >= 2 and count / total >= 0.5:
            return "partial"
        return "none"

    # ------------------------------------------------------------------
    # Normalization / parsing
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        """Normalize an answer to an equivalence key."""
        s = text.strip().lower()
        # Numbers: strip $ , and trailing % for equivalence of 31.5 vs 31.5
        cleaned = s.replace("$", "").replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        # Try numeric conversion
        try:
            num = float(cleaned)
            if num == int(num):
                return str(int(num))
            return f"{num:.6f}".rstrip("0").rstrip(".")
        except (ValueError, TypeError):
            pass
        # Booleans
        if s in ("true", "false"):
            return s
        return s

    def _is_numeric(self, text: str) -> bool:
        try:
            float(self._normalize(text))
            return True
        except (ValueError, TypeError):
            return False

    # ------------------------------------------------------------------
    # Confidence extraction from raw model output
    # ------------------------------------------------------------------
    @staticmethod
    def extract_reported_confidence(raw: str, default: float = 0.5) -> float:
        """Best-effort parse of a model's self-reported confidence."""
        if not raw:
            return default
        # Look for "CONFIDENCE: X" anywhere in the output (case-insensitive).
        m = re.search(r"confidence\s*[:=]\s*(0?\.\d+|[01](?:\.0*)?)", raw, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        # Look for a standalone leading number between 0 and 1.
        m2 = re.match(r"\s*(0?\.\d+|[01](?:\.0*)?)\s*", raw)
        if m2:
            try:
                val = float(m2.group(1))
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
        return default
