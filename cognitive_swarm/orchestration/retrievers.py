"""Pluggable retriever layer behind the truth router's Tier 3.

The production design (docs/ARCHITECTURE.md §5) is a set of self-contained
connectors, one per external source. Each connector turns a question into a
list of ``SourceClaim`` objects with a reliability and an independence flag,
so the ``Corroborator`` can do reliability-weighted agreement across sources —
including honest conflict reporting when reliable sources disagree.

The layer is intentionally *media-agnostic*: a ``Retriever`` does not care
whether its underlying source is text documents, a database, a web API, or a
built-in interpreter. It just returns claims. This means a future multimodal
model can consume images/PDFs/tables through the same interface with no rework
(see docs/ARCHITECTURE.md §7).

This module ships two implementations:

  * ``TableRetriever``    — a simple hard-coded knowledge table (used to
                            demonstrate the corroboration mechanism and to keep
                            the historical demo deterministic).
  * ``LocalDocsRetriever`` — a lightweight, dependency-free retriever that
                            scans a directory of text documents, chunks them,
                            and returns the passages most relevant to the
                            question by stoplist-filtered lexical overlap.
                            This is the minimal real connector that turns Tier 3
                            from a stub into a working retrieval step.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from cognitive_swarm.memory.verified_memory import STOPWORDS
from cognitive_swarm.orchestration.corroboration import SourceClaim
from cognitive_swarm.orchestration.prompt_optimizer import TaskType


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z][a-z0-9']*", text.lower())


def _significant(tokens: List[str]) -> List[str]:
    """Drop generic words that carry no discrimination for retrieval."""
    return [t for t in tokens if t not in STOPWORDS]


class Retriever(ABC):
    """A single connector from a question to candidate source claims.

    Each connector declares which question *categories* it is relevant to via
    ``categories`` (a list of ``TaskType``). ``None`` means "applies to any
    question". The router's Tier 3 uses this as a lightweight **gate**: it only
    queries a connector when the question's category is in the connector's
    list. This is the "path of least resistance" filter — a code question never
    triggers a WebRetriever lookup, an arithmetic question never scans local
    history docs — which keeps retrieval cheap and keeps irrelevant, verbose
    claims out of corroboration.
    """

    name: str = "retriever"
    categories: Optional[Sequence[TaskType]] = None  # None = any category

    def applies_to(self, task_type: Optional[TaskType]) -> bool:
        """True if this connector should be queried for the given category."""
        if not self.categories:
            return True
        return task_type in self.categories

    @abstractmethod
    def get_claims(self, question: str) -> List[SourceClaim]:
        """Return the claims this source supports for the question."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.name!r} categories={self.categories}>"


class TableRetriever(Retriever):
    """A tiny hand-authored knowledge table.

    Kept for illustration and for the deterministic corroboration demo. In a
    real system this generalises to any API/store that returns keyed facts.
    """

    def __init__(self, table: Dict[str, List[dict]], name: str = "reference-table",
                 categories: Optional[Sequence[TaskType]] = None):
        self.name = name
        self.table = table
        self.categories = categories

    def get_claims(self, question: str) -> List[SourceClaim]:
        claims: List[SourceClaim] = []
        norm = question.strip().lower()
        for key, sources in self.table.items():
            if key in norm:
                for s in sources:
                    claims.append(SourceClaim(
                        source=s["name"], answer=s["answer"],
                        reliability=s.get("reliability", 0.5),
                        independent=s.get("independent", True),
                        reason=s.get("reason", ""),
                    ))
        return claims


class LocalDocsRetriever(Retriever):
    """Scan a folder of text documents and return relevant passages as claims.

    A minimal, dependency-free RAG-style connector: read all ``.txt`` files
    under ``source_dir``, split them into passage chunks of ``chunk_size``
    sentences, score each chunk against the question by stoplist-filtered
    token overlap, and emit the single best concrete passage as one
    ``SourceClaim`` (via ``_extract_answer``).

    Chunk provenance (file + approximate offsets) is carried in ``source`` and
    ``reason`` so corroboration can trace where an answer came from.

    Parameters
    ----------
    source_dir : str | os.PathLike
        Directory tree of text documents to index (recursively scanned).
    name : str
        Label used as the source prefix in emitted claims.
    chunk_size : int
        Number of sentences per passage chunk.
    match_threshold : float
        Minimum lexical overlap (0..1) for a passage to be considered a hit.
    reliability : float
        Authority assigned to this document collection.
    categories : sequence of TaskType, optional
        Task-type categories this retriever serves; used for gating.
    """

    def __init__(self, source_dir, name: str = "local-docs",
                 chunk_size: int = 3,
                 match_threshold: float = 0.20, reliability: float = 0.8,
                 categories: Optional[Sequence[TaskType]] = None):
        self.name = name
        self.source_dir = Path(source_dir)
        self.chunk_size = chunk_size
        self.match_threshold = match_threshold
        self.reliability = reliability
        self.categories = categories
        self._index = self._build_index()

    # ------------------------------------------------------------------
    # Index construction (load once at construction, cheap at query time).
    # ------------------------------------------------------------------
    def _build_index(self) -> List[dict]:
        """Return a list of {source, text, sig, path, offset} passage records."""
        passages: List[dict] = []
        if not self.source_dir.exists():
            return passages
        for path in sorted(self.source_dir.rglob("*.txt")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, chunk in enumerate(self._chunk_sentences(text, self.chunk_size)):
                sig = _significant(_tokens(chunk["text"]))
                if not sig:
                    continue
                passages.append({
                    "source": f"{self.name}:{path.relative_to(self.source_dir)}",
                    "text": chunk["text"],
                    "sig": set(sig),
                    "path": str(path),
                    "offset": chunk["start"],
                })
        return passages

    @staticmethod
    def _chunk_sentences(text: str, chunk_size: int) -> List[dict]:
        """Split text into passage chunks, each up to ~chunk_size sentences.

        Real documents are not one long sentence stream: they contain section
        headers ("Topic: ...") and paragraphs separated by blank lines. In
        reference notes the header carries the *topic words* and the paragraph
        carries the *answer*, so we MERGE each header with its following
        paragraph into one chunk — lexical retrieval then matches the topic and
        ``_extract_answer`` still finds the fact. Remaining long paragraphs are
        windowed to ``chunk_size`` sentences.
        """
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        merged: List[str] = []
        i = 0
        while i < len(paras):
            if LocalDocsRetriever._looks_like_header(paras[i]) and i + 1 < len(paras) \
                    and not LocalDocsRetriever._looks_like_header(paras[i + 1]):
                merged.append(f"{paras[i]} {paras[i + 1]}")
                i += 2
            else:
                merged.append(paras[i])
                i += 1
        chunks: List[dict] = []
        for para in merged:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            if len(sentences) <= chunk_size:
                chunks.append({"text": para, "start": len(chunks)})
                continue
            for j in range(0, len(sentences), chunk_size):
                pieces = " ".join(s.strip() for s in sentences[j:j + chunk_size] if s.strip())
                if pieces:
                    chunks.append({"text": pieces, "start": len(chunks)})
        return chunks

    @staticmethod
    def _looks_like_header(para: str) -> bool:
        low = para.strip().lower()
        return low.startswith("topic:") or low.startswith("reference notes") \
            or low.startswith("parameter") or low.startswith("chapter")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_claims(self, question: str) -> List[SourceClaim]:
        q_sig = _significant(_tokens(question))
        if not q_sig:
            return []
        q_set = set(q_sig)
        scored: List[tuple] = []
        for p in self._index:
            if not p["sig"]:
                continue
            inter = len(q_set & p["sig"])
            # Score by query token recall: how much of the question this passage
            # covers. Normalising by the query (not the passage) keeps long
            # passages from being unfairly penalised, so any document that
            # actually addresses the question ranks high.
            score = inter / len(q_set)
            if score >= self.match_threshold:
                scored.append((score, p))
        if not scored:
            return []
        scored.sort(key=lambda t: t[0], reverse=True)

        # One connector = one source voice. A single connector should return a
        # single, self-consistent best claim; CORROBORATION is what happens
        # ACROSS connectors (this org, a table, a web source). Returning many
        # top passages here would feed mutually-inconsistent extracts into the
        # corroborator and manufacture false conflicts. Prefer the best passage
        # that yields a concrete extracted fact (not a template header).
        best = max(scored, key=lambda t: (self._is_concrete(t[1]), t[0]))
        score, p = best
        return [SourceClaim(
            source=p["source"],
            answer=self._extract_answer(p),
            reliability=min(self.reliability + score * 0.1, 1.0),
            independent=True,
            reason=f"passage at {p['path']} (offset {p['offset']}, overlap {score:.2f})",
        )]

    @staticmethod
    def _is_concrete(passage: dict) -> bool:
        """True if a passage carries a concrete extracted fact rather than a
        template/header clause (headers rarely contain the actual answer)."""
        return not LocalDocsRetriever._extract_answer(passage).lower().startswith(
            ("topic:", "reference notes", "parameter"))

    @staticmethod
    def _extract_answer(passage: dict) -> str:
        """Pull a concise, comparable fact out of a retrieved passage.

        Corroboration groups claims by their *answer*, so a passage must yield
        the same short fact a table entry would ("476 CE", "H2O") rather than
        its full prose, or two sources that actually agree would look like a
        conflict. This lightweight, dependency-free extractor recognises the
        fact shapes common in a small reference corpus:

          * a year with an era suffix  -> "476 CE", "1096"
          * an ERA code                -> "H2O" (letters+digits, 2..6 chars)
          * a temperature              -> "0 degrees Celsius"
          * a key quantity (commas)    -> "299,792,458 metres per second"

        Anything else falls back to the passage's opening clause.
        """
        text = passage["text"]
        m = re.search(r"\b(\d{1,4}\s*(?:CE|AD|BCE|BC)\b)", text)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        m = re.search(r"\b(1[0-9]{3}|[0-9]{4})\b", text)
        if m:
            return m.group(1)
        # Chemical formula / compact code: a token starting with a capital
        # letter that CONTAINS at least one digit, e.g. "H2O", "H2", "CaCl2".
        # Requiring a digit skips header words like "Topic" or "Reference".
        m = re.search(r"\b([A-Z][A-Za-z0-9]*\d[A-Za-z0-9]*)\b", text)
        if m and 2 <= len(m.group(1)) <= 6:
            return m.group(1)
        m = re.search(r"(-?\d+)\s*degrees?\s*(Celsius|Fahrenheit)", text)
        if m:
            return f"{m.group(1)} degrees {m.group(2)}"
        m = re.search(r"(\d{1,3}(?:,\d{3})+)\s+([a-z ]{1,20})", text)
        if m:
            return f"{m.group(1)} {m.group(2)}".strip()
        # Fall back to the passage's opening clause, trimmed.
        return re.split(r"(?<=[.!?])\s+", text)[0].strip()
