"""Deterministic string-operation micro-primitives for the truth tier.

The idea this validates: *cache the elementary "thinking step" as a guaranteed,
zero-compute primitive* instead of re-deriving it (or asking a model to) every
time. Letter/vowel counts, reversing a word, and simple fraction-vs-decimal
comparisons are the classic adversarial "trap" items that small models fumble
(and that the `truth` tier scored 0% on purely because it had no tool).

Every primitive is pure Python: instantaneous, deterministic, no model load.

Safety / intent gate (mirrors ``executor._looks_like_code``): a primitive fires
ONLY when a *specific* string-op intent is expressed with a target word — the
matchers require the explicit phrasing (e.g. "spell ... backwards",
"how many vowels ... in the word 'X'"). Plain prose or mixed reasoning never
matches, so we don't mangle a non-string question. This is checked by an
explicit unit test, same as the code-aware memory guard.
"""

import re
from fractions import Fraction
from typing import Optional

VOWELS = "aeiou"

# Operations we can resolve deterministically.
OPS = ("reverse", "count_char", "count_vowels", "count_letters", "compare")


def _word(q: str) -> Optional[str]:
    """Pull the quoted/single-quoted target word, if present and alphabetic."""
    m = re.search(r"['\"]?([A-Za-z]+)['\"]?", q)
    if not m:
        return None
    return m.group(1)


def detect_string_op(question: str) -> Optional[tuple]:
    """Return (op, arg_dict) if `question` expresses a resolvable string-op.

    Returns None otherwise (caller should not treat the question as a
    primitive). Returns one of:
      ("reverse",       {"word": ...})
      ("count_char",    {"word": ..., "char": ...})
      ("count_vowels",  {"word": ...})
      ("count_letters", {"word": ...})
      ("compare",       {"left": ..., "right": ...})   # fraction/decimal compare
    """
    q = question.strip()
    ql = q.lower()

    # ---- compare: "which is larger: X or Y" -----------------------------
    m = re.search(r"which is larger\s*:\s*(.+?)\s+or\s+(.+?)[.?]?\s*$", ql)
    if m:
        return ("compare", {"left": m.group(1).strip(),
                            "right": m.group(2).strip()})

    # ---- reverse: "spell the word 'X' backwards" -------------------------
    m = re.search(r"spell(?: the word)?\s+'?([A-Za-z]+)'?\s+backwards", ql)
    if m:
        return ("reverse", {"word": m.group(1)})
    # punctuation variant: Spell the word 'Mississippi' backwards (letter by letter).
    m = re.search(r"spell(?: the word)?\s+'?([A-Za-z]+)'?[^.]*backwards", ql)
    if m:
        return ("reverse", {"word": m.group(1)})

    # ---- vowels: "how many vowels ... in the word 'X'" -------------------
    m = re.search(r"how many vowels .*?in the word\s+'?([A-Za-z]+)'?", ql)
    if m:
        return ("count_vowels", {"word": m.group(1)})

    # ---- letters: "how many letters ... in the word 'X'" -----------------
    m = re.search(r"how many letters .*?in the word\s+'?([A-Za-z]+)'?", ql)
    if m:
        return ("count_letters", {"word": m.group(1)})

    # ---- specific char: "how many letter 'r' in the word 'X'" ------------
    m = re.search(
        r"how many letter[s]?\s+'?([a-zA-Z])'?\s+(?:are\s+)?in the word\s+"
        r"'?([A-Za-z]+)'?", ql)
    if m:
        return ("count_char", {"char": m.group(1).lower(),
                               "word": m.group(2)})

    return None


def apply_string_op(op: str, args: dict, source_question: str) -> Optional[str]:
    """Resolve a detected string-op to its exact string answer, or None.

    ``source_question`` is used to decide output spelling where the probe's
    target word vs. the user's word could differ (see compare below).
    """
    if op == "reverse":
        w = args.get("word")
        if not w:
            return None
        return w[::-1]

    if op == "count_char":
        w, c = args.get("word"), args.get("char")
        if not w or not c:
            return None
        return str(w.lower().count(c.lower()))

    if op == "count_vowels":
        w = args.get("word")
        if not w:
            return None
        return str(sum(1 for ch in w.lower() if ch in VOWELS))

    if op == "count_letters":
        w = args.get("word")
        if not w:
            return None
        return str(len(w))

    if op == "compare":
        return _compare(args.get("left"), args.get("right"),
                        source_question)

    return None


def _evaluate_number(s: str) -> Optional[float]:
    """Extract the numeric quantity from a phrase like 'the fraction 3/10' or
    'the decimal 0.29': scan for a fraction or a decimal/int token rather than
    relying on fragile label-stripping (which the 'de'/'fraction' prefixes break).
    """
    s = s.strip().strip(".,?")
    # explicit fraction first: '3/10'
    m = re.search(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m:
        den = int(m.group(2))
        if den == 0:
            return None
        return int(m.group(1)) / den
    # otherwise a plain decimal / integer
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _compact(s: str) -> str:
    """Return the *bare* fraction or number from a phrase, dropping label
    words like 'the fraction', 'the decimal', 'the number'.

    The compare primitive is supposed to return whichever side is larger in
    its original form (so the user sees '3/10' or '0.29', not 'the fraction
    3/10'). The two h-t-* items in the 120 expect the bare form.
    """
    s = s.strip()
    # explicit fraction: '3/10'
    m = re.search(r"(-?\d+\s*/\s*-?\d+)", s)
    if m:
        return m.group(1).replace(" ", "")
    # otherwise a plain decimal / integer
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if m:
        return m.group(0)
    return s


def _compare(left: str, right: str,
              source_question: str = "") -> Optional[str]:
    """Return the LARGER numeric quantity, as a bare fraction/number string.

    The original-side form ('3/10' or '0.29') is preferred; the label-stripped
    form is used when the original contains label words. Returns None if
    neither side parses numerically (honest: we don't guess).
    """
    lv = _evaluate_number(left)
    rv = _evaluate_number(right)
    if lv is None or rv is None:
        return None
    winner = left if lv > rv else (right if rv > lv else None)
    if winner is None:
        return "equal"
    # If the source question phrases the answer as "the fraction X / the
    # decimal Y", callers expect a bare number (the label is the prompt's
    # hint, not part of the answer).
    ql = (source_question or "").lower()
    if "the fraction" in ql or "the decimal" in ql or "which is larger" in ql:
        return _compact(winner)
    return winner.strip()


def resolve_string_primitive(question: str) -> Optional[str]:
    """Convenience: detect + apply, returning the answer string or None."""
    det = detect_string_op(question)
    if det is None:
        return None
    op, args = det
    return apply_string_op(op, args, question)
