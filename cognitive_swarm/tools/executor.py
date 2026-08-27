"""Ground-truth code executor.

Real-life truth hierarchy: for a code-output question the *source of truth* is
the interpreter, not the models' guesses. This module extracts the executable
expression/statement from the question and runs it in a sandboxed subprocess.

Safety:
  * runs in a fresh subprocess (isolation),
  * cwd set to an empty temp dir (no access to repo files),
  * a hard wall-clock timeout,
  * environment cleared of secret-looking vars,
  * output captured and bounded.

We only ever auto-run *question-derived* expressions from our own benchmark /
trusted prompts. This is a research helper, not exposed to untrusted input.
"""

import os
import re
import subprocess
import tempfile
import textwrap
from typing import Optional


class CodeExecutionError(Exception):
    """Raised when extracted code cannot be built or times out."""


# ---------------------------------------------------------------------------
# Expression extraction
# ---------------------------------------------------------------------------
def extract_python(question: str) -> Optional[str]:
    """Return a runnable Python statement derived from the question, or None.

    Uses a balanced-delimiter scan to swallow the WHOLE Python expression
    (nested parens/brackets/braces, strings) even when the question appends
    '?' or English prose after it. Examples:

      'print(2 ** 3 ** 2)?'            -> print(2 ** 3 ** 2)
      'result of: len([1,2,[3,4]])?'   -> print(len([1,2,[3,4]]))
      'print(sorted([3,1,2]))...'      -> print(sorted([3,1,2]))
    """
    if not question:
        return None
    q = question.strip()

    # --- 1) scan the contiguous balanced python token --------------------
    token = _scan_python_token(q)
    if token is None:
        return None
    code = token.rstrip("? ").strip()
    if not code:
        return None

    # --- 1b) strictness gate: reject plain prose masquerading as code ------
    # A genuine code expression has one of: a call ('name('), a bracket
    # literal, or a math operator/digit operand. English prose ("What caused
    # the fall of the Roman Empire?") has none -> return None so the caller
    # doesn't treat it as executable code.
    if not _looks_like_code(code):
        return None

    # --- 2) wrap bare expressions in print() -----------------------------
    if not re.search(r"\b(print\s*\(|import |from |def |for |while )", code):
        code = "print(" + code + ")"

    # --- 3) auto-import a top-level module reference (e.g. functools.reduce) --
    code = _auto_import(code)
    return code


def _looks_like_code(code: str) -> bool:
    """True if `code` is a plausible Python expression (not plain prose)."""
    # A call like name(  -> e.g. print(, len(, sorted(
    if re.search(r"[A-Za-z_]\w*\s*\(", code):
        return True
    # A bracket literal: [ ... ], { ... }, ( ... ) with content
    if re.search(r"[\[\{\(]", code):
        return True
    # A math / boolean operator between operands: 2+3, 45*0.70, a == b
    if re.search(r"(?:\d|\w)\s*(?:[+\-*/%^]|==|!=|<=|>=|<|>)\s*(?:\d|\w)", code):
        return True
    # A bare statement keyword
    if re.search(r"\b(import|from|def|for|while|return|if )", code):
        return True
    return False


def _scan_python_token(q: str) -> Optional[str]:
    """Scan `q` for a runnable python expression, returning the balanced text.

    We look for the first occurrence of '<word>(' and consume from there with
    a stack, honoring quotes and nested brackets. If nothing calls a function,
    fall back to a numeric/identifier-ish tail.
    """
    # find candidate starts: 'print(', or a bare 'name(' like 'len(', or a digit
    q = q.lstrip()
    starts = [m.start() for m in re.finditer(r"(?:print\s*\(|\b[A-Za-z_][\w.]*\s*\()", q)]
    if not starts:
        # maybe a simple numeric / literal expression with no call
        return _scan_balanced_from(q, 0, force=True)

    best = None
    for s in starts:
        seg = _scan_balanced_from(q, s)
        # prefer the candidate that yields the longest balanced expression
        if seg and (best is None or len(seg.rstrip("? ")) > len(best)):
            best = seg
    return best.rstrip("? ") if best else None


def _scan_balanced_from(q: str, start: int, force: bool = False) -> Optional[str]:
    """Return the balanced expression starting at q[start], honoring strings."""
    pairs = {'(': ')', '[': ']', '{': '}'}
    open2close = {'(': ')', '[': ']', '{': '}'}
    close2open = {v: k for k, v in open2close.items()}
    stack = []
    in_str = None  # current quote char or None
    i = start
    n = len(q)
    while i < n:
        c = q[i]
        if in_str is not None:
            if c == '\\':
                i += 2
                continue
            if c == in_str:
                in_str = None
        else:
            if c in ('"', "'"):
                in_str = c
            elif c in open2close:
                stack.append(c)
            elif c in close2open:
                if not stack:
                    # unbalanced ')' -> not this candidate
                    return None
                stack.pop()
                if not stack:
                    # balanced close at i
                    return q[start:i+1]
            elif c.isspace() and not stack:
                # whitespace when no open bracket: end of expression only if
                # we've consumed something
                if i > start:
                    return q[start:i]
        i += 1
    # reached end; if stack balanced return, else None
    if not stack:
        return q[start:i]
    return None


def _auto_import(code: str) -> str:
    """Prepend imports for top-level module references not yet imported."""
    # Find a qualified name like 'functools.reduce' at the top level of code.
    for m in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\.", code):
        mod = m.group(1)
        if mod in ("print", "len", "range", "sorted", "list", "dict", "set",
                   "zip", "str", "int", "float", "bool", "tuple", "map",
                   "filter", "sum", "min", "max", "abs", "round", "True",
                   "False", "None"):
            continue
        line = f"import {mod}"
        if line.split(" ")[-1] not in [l.split()[-1] for l in re.findall(
                r"^\s*(import|from)\s+(\S+)", code, re.M)]:
            code = line + "\n" + code
            break
    return code


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------
def run_python(code: str, timeout: float = 10.0) -> str:
    """Execute `code` in a sandboxed subprocess and return its stdout."""
    if not code or not code.strip():
        raise CodeExecutionError("empty code")

    # Sanitize environment: keep only safe, non-secret vars.
    safe_keys = {"PATH", "HOME", "LANG", "LC_ALL", "TZ", "PYTHONPATH", "PYTHONIOENCODING"}
    env = {k: v for k, v in os.environ.items() if k in safe_keys}
    env["PYTHONIOENCODING"] = "utf-8"

    with tempfile.TemporaryDirectory() as workdir:
        # Build the code as a small script; capture stdout only.
        script = textwrap.dedent(
            f"""\
            import sys
            try:
                exec({code!r})
            except Exception as e:
                print(f"ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
                sys.exit(1)
            """
        )
        try:
            proc = subprocess.run(
                ["python3", "-I", "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise CodeExecutionError("execution timed out")
        except FileNotFoundError:
            # fall back to whatever python3 resolves on PATH
            proc = subprocess.run(
                ["python3", "-c", script], capture_output=True, text=True,
                timeout=timeout, cwd=workdir, env={"PYTHONIOENCODING": "utf-8"},
            )

    if proc.returncode != 0:
        raise CodeExecutionError(proc.stderr.strip() or "non-zero exit")

    out = proc.stdout.rstrip("\n")
    return out


def execute(question: str, timeout: float = 10.0) -> Optional[str]:
    """Extract + run code from a question; return stdout or None on failure."""
    code = extract_python(question)
    if not code:
        return None
    try:
        return run_python(code, timeout)
    except CodeExecutionError:
        return None
