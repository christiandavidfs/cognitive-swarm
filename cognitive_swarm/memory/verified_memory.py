"""Verified memory: the swarm's accumulating, provenance-aware 'experience'.

Real-life learning has two rules that this store encodes:

1. You don't recompute what you already know -> persistent memory with fast
   lookup (exact + robust similarity that IGNORES boilerplate words).

2. Knowledge carries provenance and confidence. Different domains have
   different sources of truth (code = interpreter, math = calculator, history
   = primary sources that can DISAGREE). So each entry stores not just an
   answer but:
       answer        : the concluded value
       confidence    : 0..1 how sure we are
       sources       : list of {name, reliability} that support it
       reliability   : highest reliability among corroborating sources
       disagreement  : list of {source, answer, reason} for conflicting views
       tier          : how it was established (executor/calculator/retrieval/
                       debate/manual)

For contested domains (history, opinions, ambiguous chemistry) we NEVER force
a single number — we preserve disagreement so the system reports "sources
disagree" honestly instead of faking certainty.

Lookup similarity uses a STOPLIST-filtered token overlap so that the shared
boilerplate of code questions ("what does this python code output:") does not
cause two DIFFERENT expressions to look the same.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_MEMORY_PATH = Path(__file__).parent.parent.parent / "verified_memory.json"

# Generic words that must not influence similarity (they carry no signal).
STOPWORDS = {
    "what", "does", "this", "the", "a", "an", "is", "of", "in", "on", "to",
    "and", "or", "for", "python", "code", "output", "print", "result", "with",
    "that", "from", "by", "its", "it", "if", "you", "can", "tell", "me",
    "about", "how", "much", "many", "which", "when", "who", "are", "was",
    "were", "be", "their", "they", "not", "do", "does", "did", "these", "those",
    "please", "question", "answer", "solve", "given", "following", "below",
}


def _significant(tokens: List[str]) -> List[str]:
    """Drop boilerplate tokens that provide no discrimination."""
    return [t for t in tokens if t not in STOPWORDS]


class VerifiedMemory:
    """Persistent, provenance-aware store of verified knowledge."""

    def __init__(self, path: Optional[os.PathLike] = None, similarity_threshold: float = 0.75):
        self.path = Path(path) if path else DEFAULT_MEMORY_PATH
        self.similarity_threshold = similarity_threshold
        self.entries: Dict[str, dict] = {}
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if isinstance(data, dict):
                    self.entries = data
            except (ValueError, OSError):
                self.entries = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def remember(
        self,
        question: str,
        answer: str,
        source: str = "ground-truth",
        tier: str = "executor",
        confidence: float = 1.0,
        reliability: float = 1.0,
        sources: Optional[List[dict]] = None,
        disagreement: Optional[List[dict]] = None,
        verified: bool = True,
    ) -> None:
        """Archive a verified (question -> answer) record with provenance."""
        key = self.normalize(question)
        if not key or not answer:
            return
        if sources is None:
            sources = [{"name": source, "reliability": reliability}]
        if disagreement is None:
            disagreement = []
        existing = self.entries.get(key)
        self.entries[key] = {
            "question": question,
            "answer": answer,
            "tier": tier,
            "confidence": float(confidence),
            "reliability": float(reliability),
            "sources": sources,
            "disagreement": disagreement,
            "code_sig": self._code_sig(question),
            "string_op_sig": self._string_op_sig(question),
            "verified": verified,
            "hits": (existing["hits"] + 1 if existing else 0),
        }
        self.save()

    def bump(self, question: str) -> None:
        key = self.normalize(question)
        if key in self.entries:
            self.entries[key]["hits"] += 1
            self.save()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def _code_sig(self, text: str) -> Optional[str]:
        """Normalized code signature for a question containing executable code,
        or None if it's not code-like."""
        from cognitive_swarm.tools.executor import extract_python
        code = extract_python(text)
        if not code:
            return None
        # Normalize whitespace and strip the leading print( for comparison so
        # 'print(2+3)' matches '{2+3}'-style rephrases consistently.
        return re.sub(r"\s+", "", code)

    def _string_op_sig(self, text: str) -> Optional[str]:
        """Signature for a question that expresses a resolvable string-op
        (reverse/count/compare), used so near-duplicate string questions about
        DIFFERENT words never blur together in memory.

        Mirrors the code-aware guard: instead of stoplist-filtered token overlap
        (which would wrongly merge 'spell strawberry backwards' with 'spell
        mississippi backwards'), we require an IDENTICAL (op, targets) signature."""
        from cognitive_swarm.tools.string_primitives import detect_string_op
        det = detect_string_op(text)
        if not det:
            return None
        op, args = det
        return op + ":" + "|".join(str(v) for _, v in sorted(args.items()))

    def lookup(self, question: str) -> Optional[dict]:
        """Return the stored record (full provenance) for this question or a
        close, salient rephrase. Returns None if no strong match.

        DISTINCT code questions never blur together: if the query contains
        executable code, we require the stored entry to have the SAME code
        signature (soft token-overlap would wrongly merge e.g. sorted([1,2,3])
        and len([1,2,[3,4]]) because they share digits)."""
        key = self.normalize(question)
        if key and key in self.entries:
            self.bump(question)
            return self.entries[key]

        # Code-aware path: exact normalized-code identity, else no match.
        q_code = self._code_sig(question)
        if q_code is not None:
            for entry in self.entries.values():
                stored_sig = (entry.get("code_sig") or self._code_sig(entry["question"]))
                if stored_sig and stored_sig == q_code:
                    self.bump(question)
                    return entry
            return None

        # String-op-aware path: identical (op, targets) signature, else no match.
        q_so = self._string_op_sig(question)
        if q_so is not None:
            for entry in self.entries.values():
                stored_so = (entry.get("string_op_sig") or self._string_op_sig(entry["question"]))
                if stored_so and stored_so == q_so:
                    self.bump(question)
                    return entry
            return None

        # Non-code: stoplist-filtered token overlap.
        q_sig = _significant(self._tokens(question))
        if not q_sig:
            return None
        best, best_score = None, 0.0
        for entry in self.entries.values():
            score = self._score(q_sig, _significant(self._tokens(entry["question"])))
            if score > best_score:
                best, best_score = entry, score
        if best_score >= self.similarity_threshold:
            self.bump(question)
            return best
        return None

    def get_answer(self, question: str) -> Optional[str]:
        rec = self.lookup(question)
        return rec["answer"] if rec else None

    def size(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def normalize(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def _tokens(s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    def _score(self, a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        set_b = set(b)
        inter = sum(1 for t in a if t in set_b)
        return inter / max(len(a), len(b), 1)
