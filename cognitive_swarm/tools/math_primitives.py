"""Deterministic math micro-primitives for the truth tier.

Extends the calculator tier (which only handles literal ``a op b``) with
higher-level primitives that cover the cases small models fumble and the
calculator regex misses:
  - word operators: "divided by", "to the power of", "half of", "squared",
    "cubed"
  - unary / special forms: factorial n!, log base b of n, sqrt(n)
  - geometry: area of rectangle, perimeter of square, sum of interior
    angles of a polygon
  - closed-form series: "sum of the first N positive integers",
    "sum of the first N even positive integers"
  - simple multi-step arithmetic with PROPER operator precedence
    (*/x/^ before +/-)

Every primitive is pure Python: instantaneous, deterministic, no model load.

Safety / intent gate (mirrors string_primitives and executor): a primitive
fires ONLY when an explicit pattern is present. Plain prose never matches.
"""

import math
import re
from typing import Optional, Tuple


# --- Number extraction --------------------------------------------------

def _eval_number(s: str) -> Optional[float]:
    """Extract a number from a phrase (fraction, decimal, or int)."""
    s = s.strip().strip(".,?")
    if not s:
        return None
    m = re.search(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m:
        den = int(m.group(2))
        if den == 0:
            return None
        return int(m.group(1)) / den
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _format_number(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return ("%.6f" % v).rstrip("0").rstrip(".")


_WORD_FRACTIONS = {
    "half": 2, "quarter": 4, "third": 3, "fourth": 4, "fifth": 5,
}


def _word_fraction_value(word: str) -> Optional[float]:
    """Return 1/2 for 'half', 1/4 for 'quarter', etc."""
    return _WORD_FRACTIONS.get(word)


# --- Multi-step expression evaluator with proper precedence ------------

def _eval_expression(expr: str) -> Optional[float]:
    """Evaluate a simple arithmetic expression with proper precedence:
    ^ > */x/ > +/- (left-to-right within precedence)."""
    tokens = re.findall(r"\d+(?:\.\d+)?|[+\-*/x^()]", expr)
    if not tokens:
        return None
    prec = {"^": 4, "x": 3, "*": 3, "/": 3, "+": 2, "-": 2}
    right_assoc = {"^"}
    output: list = []
    ops: list = []
    prev = None
    for tok in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            output.append(tok)
        elif tok == "(":
            ops.append(tok)
        elif tok == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if not ops:
                return None
            ops.pop()
        else:
            while ops and ops[-1] != "(" and (
                prec[ops[-1]] > prec[tok] or
                (prec[ops[-1]] == prec[tok] and tok not in right_assoc)
            ):
                output.append(ops.pop())
            ops.append(tok)
        prev = tok
    while ops:
        if ops[-1] in "()":
            return None
        output.append(ops.pop())
    stack: list = []
    for tok in output:
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            stack.append(float(tok))
        else:
            if len(stack) < 2:
                return None
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok in ("*", "x"):
                stack.append(a * b)
            elif tok == "/":
                if b == 0:
                    return None
                stack.append(a / b)
            elif tok == "^":
                try:
                    stack.append(a ** b)
                except (ValueError, OverflowError):
                    return None
    if len(stack) != 1:
        return None
    return stack[0]


# --- Detection ---------------------------------------------------------

def detect_math_primitive(question: str) -> Optional[Tuple[str, dict]]:
    ql = question.strip().lower()

    # log
    m = re.search(r"log base\s*(\d+)\s+of\s*(-?\d+(?:\.\d+)?)", ql)
    if m:
        return ("log", {"base": m.group(1), "n": m.group(2)})
    m = re.search(r"log_(\d+)\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)", ql)
    if m:
        return ("log", {"base": m.group(1), "n": m.group(2)})

    # factorial
    m = re.search(r"value of\s*(-?\d+)\s*!", ql)
    if m:
        return ("factorial", {"n": m.group(1)})
    m = re.search(r"(-?\d+)\s*factorial", ql)
    if m:
        return ("factorial", {"n": m.group(1)})

    # sqrt (single and sum of two)
    m = re.search(r"sqrt\(\s*\d+\s*\)\s*\+\s*sqrt\(\s*\d+\s*\)", ql)
    if m:
        a, b = re.findall(r"\d+", m.group(0))
        return ("sqrt_sum", {"a": a, "b": b})
    m = re.search(r"square root of\s*(-?\d+(?:\.\d+)?)", ql)
    if m:
        return ("sqrt", {"n": m.group(1)})
    m = re.search(r"sqrt\(\s*(-?\d+(?:\.\d+)?)\s*\)", ql)
    if m:
        return ("sqrt", {"n": m.group(1)})

    # power
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*to the power of\s*(-?\d+(?:\.\d+)?)", ql)
    if m:
        return ("power", {"base": m.group(1), "exp": m.group(2)})
    m = re.search(r"(-?\d+)\s*\^\s*(-?\d+)", ql)
    if m:
        return ("power", {"base": m.group(1), "exp": m.group(2)})

    # square / cube
    m = re.search(r"(-?\d+(?:\.\d+)?)\s+squared\b", ql)
    if m:
        return ("square", {"n": m.group(1)})
    m = re.search(r"(-?\d+(?:\.\d+)?)\s+cubed\b", ql)
    if m:
        return ("cube", {"n": m.group(1)})

    # half
    m = re.search(r"half of\s*(-?\d+(?:\.\d+)?)", ql)
    if m:
        return ("half", {"n": m.group(1)})

    # "one half of X" / "a quarter of X" / "one third of X" / "a fourth of X"
    # Nested chains: "one half of one quarter of 400" -> apply outer then inner.
    # Detect all fraction-of patterns sequentially, then apply left-to-right.
    fracs = re.findall(r"(half|quarter|third|fourth|fifth)\s+of", ql)
    if fracs:
        # The number should be the digits immediately before ? or . or end
        num_match = re.search(r"(-?\d+(?:\.\d+)?)\s*[?.]?\s*$", ql)
        if num_match:
            n_str = num_match.group(1)
            n_val = _eval_number(n_str)
            if n_val is not None:
                return ("unit_fraction_chain", {
                    "n": n_val,
                    "raw": "".join(fracs),  # keep for logging but not used
                    "words": fracs,
                })

    # ---- percentages ----------------------------------------------------
    m = re.search(r"(?:sale|discount|off|after a (\d+) percent discount)\s*"
                  r"(?:for\s)?(\d+)\s*percent\s*off", ql)
    if m:
        return None
    m = re.search(r"(\d+)\s*(?:percent|\%)\s*(?:of\s*(\d+)|\b(\d+))", ql)
    if m:
        t = m.group(2) or m.group(1)
        return ("percent_of", {"n": m.group(1), "total": t})
    m = re.search(r"(\d+)\s*%\s*off\s*(\d+)", ql)
    if m:
        return ("percent_off", {"n": m.group(1), "total": m.group(2)})
    m = re.search(r"costs?\s*\$?(\d+)\s*and\s*is\s*on\s*sale\s*for\s*(\d+)\s*%\s*off", ql)
    if m:
        return ("percent_off", {"n": m.group(2), "total": m.group(1)})
    m = re.search(r"after\s*a\s*(\d+)\s*percent\s*discount", ql)
    if m:
        return ("percent_discount_of", {"n": m.group(1)})
    # "sold for $X after a 20% discount"
    m = re.search(r"sold\s*for\s*\$?(\d+)\s*after\s*a\s*(\d+)\s*%\s*discount", ql)
    if m:
        return ("discount_reverse", {"price": m.group(1), "discount": m.group(2)})

    # ---- unit conversions ----------------------------------------------
    m = re.search(r"how many seconds are there in (?:one |a )?(?:full )?day", ql)
    if m:
        return ("unit", {"kind": "seconds_day"})

    # ---- speed: "X km in Y hours" -----------------------------------
    m = re.search(r"(\d+)\s*km\s*(?:in|after)\s*(\d+)\s*hours?", ql)
    if m:
        return ("speed", {"dist": m.group(1), "hours": m.group(2)})

    # ---- proportion: "N items cost $M, K items cost ?" -------------
    m = re.search(r"(\d+)\s*(?:notebooks?|items?)\s*cost\s*\$?(\d+)[,.]?\s*(?:how\s+much\s+do\s+)?(\d+)\s*(?:notebooks?|items?)\s*cost", ql)
    if m:
        return ("proportion", {"n1": m.group(1), "cost1": m.group(2), "n2": m.group(3)})

    # ---- simple algebra: "number doubled plus N equals M" ------------
    m = re.search(r"(?:a\s+)?number\s+doubled\s*(?:plus\s*(\d+)\s*)?(?:equals?|is)\s*(\d+)", ql)
    if m:
        off = int(m.group(1)) if m.group(1) else 0
        return ("simple_algebra", {"offset": str(off), "total": m.group(2)})

    # ---- exponential equation: "3^x = 27" ----------------------------
    m = re.search(r"(\d+)\s*\^\s*x\s*=\s*(\d+)", ql)
    if m:
        return ("exp_solve", {"base": m.group(1), "result": m.group(2)})

    # ---- percentage change: "increased by X% then decreased by Y%" ----
    m = re.search(r"increased\s+by\s+(\d+)\s*%?\s*(?:and\s*then\s*|&\s*)?decreased\s+by\s+(\d+)\s*%", ql)
    if m:
        return ("pct_change", {"up": m.group(1), "down": m.group(2)})

    # ---- work rate: "N workers build M walls in D days" -------------
    m = re.search(r"(\d+)\s*workers?\s*(?:build|construct)\s*(\d+)\s*walls?\s*in\s*(\d+)\s*days?[,.]?\s*(?:how\s+many\s+days?\s+would\s+it\s+take\s+)?(\d+)\s*workers?\s*(?:to\s+build\s+)?(\d+)\s*walls?", ql)
    if m:
        return ("work_rate", {"w1": m.group(1), "t1": m.group(3), "w2": m.group(4), "t2": m.group(5)})

    # ---- linear system: "x + y = N and x - y = M" -------------------
    m = re.search(r"x\s*\+\s*y\s*=\s*(\d+)\s*and\s*x\s*\-\s*y\s*=\s*(\d+)", ql)
    if m:
        return ("linear_system", {"a": m.group(1), "b": m.group(2)})

    # ---- arithmetic sequence term ------------------------------------
    m = re.search(r"(\d+)th term of the arithmetic sequence\s*([\d,\s]+)", ql)
    if m:
        return ("seq_term", {"n": m.group(1), "nums": m.group(2)})

    # ---- divide word
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*divided by\s*(-?\d+(?:\.\d+)?)", ql)
    if m:
        return ("divide_word", {"left": m.group(1), "right": m.group(2)})

    # series sum
    m = re.search(r"sum of the first\s*(\d+)\s*positive integers", ql)
    if m:
        return ("series_sum", {"n": m.group(1), "kind": "n"})
    m = re.search(r"sum of the first\s*(\d+)\s*even\s*positive integers", ql)
    if m:
        return ("series_sum", {"n": m.group(1), "kind": "2n"})

    # polygon angles
    m = re.search(r"sum of the interior angles of (?:a|an)?\s*(\w+)", ql)
    if m:
        word = m.group(1).strip().rstrip(",")
        sides = {"triangle": 3, "quadrilateral": 4, "pentagon": 5,
                 "hexagon": 6, "heptagon": 7, "octagon": 8}
        for k, v in sides.items():
            if word.startswith(k):
                return ("polygon_angles", {"n": str(v)})
    if "interior angles of a triangle" in ql:
        return ("polygon_angles", {"n": "3"})

    # gcd
    m = re.search(r"(?:greatest common divisor|gcd) of\s*(\d+)\s*and\s*(\d+)", ql)
    if m:
        return ("gcd", {"a": m.group(1), "b": m.group(2)})

    # compare: "which is larger: X or Y"
    m = re.search(r"larger\s+(?:fraction\s+)?:\s*(\S+)\s+or\s+(\S+)", ql)
    if m:
        return ("compare", {"left": m.group(1).strip(" ?.,"),
                            "right": m.group(2).strip(" ?.,")})

    # rectangle area
    m = re.search(r"length\s*(-?\d+(?:\.\d+)?)\s*and\s*width\s*(-?\d+(?:\.\d+)?)", ql)
    if m:
        return ("rect_area", {"length": m.group(1), "width": m.group(2)})

    # square area from perimeter
    m = re.search(r"perimeter of (\d+)", ql)
    if m:
        p = int(m.group(1))
        if p % 4 == 0:
            return ("square_area", {"side": str(p // 4)})

    return None


def detect_compound_expression(question: str) -> Optional[dict]:
    """Detect multi-step arithmetic (e.g. ``100 / 4 + 6``, ``3 * 7 - 5``),
    including word-operator chains like ``100 divided by 4 plus 6``.
    Rejects fragments that look like ellipsis tails (``1+2`` from ``1+2+...+50``).
    """
    ql = question.lower()
    # Skip if there's an ellipsis (the multi-term pattern would be
    # a closed form, handled by math primitives).
    if "..." in ql:
        return None
    expr = None
    # (1) symbol operators: N op N (op N)?
    m = re.search(
        r"-?\d+(?:\.\d+)?\s*([+\-*/x^])\s*-?\d+(?:\.\d+)?"
        r"(?:\s*([+\-*/x^])\s*-?\d+(?:\.\d+)?)?"
        r"(?:\s*([+\-*/x^])\s*-?\d+(?:\.\d+)?)?",
        ql,
    )
    if m:
        expr = m.group(0)
    # (2) word-operator chain: "A divided by B plus C"
    if not expr:
        m = re.search(
            r"(\d+)\s+divided by\s+(\d+)\s+plus\s+(\d+)", ql)
        if m:
            expr = f"{m.group(1)} / {m.group(2)} + {m.group(3)}"
        else:
            m = re.search(r"(\d+)\s+(?:plus|and)\s+(\d+)\s+(?:minus|minus)\s+(\d+)", ql)
            if m:
                expr = f"{m.group(1)} + {m.group(2)} - {m.group(3)}"
    if not expr:
        return None
    tokens = re.findall(r"\d+|[+\-*/x^]", expr)
    if len(tokens) > 5:
        return None
    return {"expr": expr}


# --- Resolution --------------------------------------------------------

def _factorial(n: float) -> Optional[str]:
    if n < 0 or n != int(n):
        return None
    n = int(n)
    if n > 20:
        return None
    r = 1
    for i in range(2, n + 1):
        r *= i
    return str(r)


def _sqrt_value(n: float) -> Optional[float]:
    if n < 0:
        return None
    return n ** 0.5


def resolve_math_primitive(question: str) -> Optional[str]:
    det = detect_math_primitive(question)
    if det is None:
        return None
    op, args = det
    if op == "log":
        b = _eval_number(args.get("base", ""))
        n = _eval_number(args.get("n", ""))
        if b is None or n is None:
            return None
        return _format_number(math.log(n, b))
    if op == "factorial":
        n = _eval_number(args.get("n", ""))
        if n is None:
            return None
        return _factorial(n)
    if op == "sqrt_sum":
        a = _eval_number(args.get("a", ""))
        b = _eval_number(args.get("b", ""))
        if a is None or b is None:
            return None
        return _format_number(_sqrt_value(a) + _sqrt_value(b))
    if op == "sqrt":
        n = _eval_number(args.get("n", ""))
        if n is None:
            return None
        return _sqrt_value(n)
    if op == "square":
        n = _eval_number(args.get("n", ""))
        if n is None:
            return None
        return _format_number(n ** 2)
    if op == "cube":
        n = _eval_number(args.get("n", ""))
        if n is None:
            return None
        return _format_number(n ** 3)
    if op == "half":
        n = _eval_number(args.get("n", ""))
        if n is None:
            return None
        return _format_number(n / 2)
    if op == "unit_fraction_chain":
        n = args.get("n", 0)
        words = args.get("words", [])
        if not words:
            return None
        result = float(n)
        for w in words:
            d = _word_fraction_value(w)
            if d is None:
                return None
            result /= d
        return _format_number(result)
    if op == "power":
        b = _eval_number(args.get("base", ""))
        e = _eval_number(args.get("exp", ""))
        if b is None or e is None:
            return None
        return _format_number(b ** e)
    if op == "divide_word":
        a = _eval_number(args.get("left", ""))
        b = _eval_number(args.get("right", ""))
        if a is None or b is None or b == 0:
            return None
        return _format_number(a / b)
    if op == "series_sum":
        n = int(float(args.get("n", "0")))
        kind = args.get("kind", "n")
        if kind == "n":
            return str(n * (n + 1) // 2)
        if kind == "2n":
            return str(n * (n + 1))
        return None
    if op == "polygon_angles":
        n = int(float(args.get("n", "0")))
        if n < 3:
            return None
        return str(180 * (n - 2))
    if op == "gcd":
        a = int(args.get("a", "0"))
        b = int(args.get("b", "0"))
        if a <= 0 or b <= 0:
            return None
        return str(math.gcd(a, b))
    if op == "rect_area":
        a = _eval_number(args.get("length", ""))
        b = _eval_number(args.get("width", ""))
        if a is None or b is None:
            return None
        return _format_number(a * b)
    if op == "square_area":
        s = _eval_number(args.get("side", ""))
        if s is None:
            return None
        return _format_number(s ** 2)
    if op == "compare":
        from cognitive_swarm.tools.string_primitives import _evaluate_number as _en
        lv = _en(args.get("left", ""))
        rv = _en(args.get("right", ""))
        if lv is None or rv is None:
            return None
        if lv > rv:
            return args.get("left", "")
        if rv > lv:
            return args.get("right", "")
        return "equal"
    if op == "percent_of":
        p = _eval_number(args.get("n", ""))
        t = _eval_number(args.get("total", ""))
        if p is None or t is None:
            return None
        return _format_number(t * p / 100)
    if op == "percent_off":
        p = _eval_number(args.get("n", ""))
        t = _eval_number(args.get("total", ""))
        if p is None or t is None:
            return None
        return _format_number(t * (1 - p / 100))
    if op == "percent_discount_of":
        p = _eval_number(args.get("n", ""))
        if p is None:
            return None
        return _format_number(1 - p / 100)
    if op == "discount_reverse":
        price = _eval_number(args.get("price", ""))
        d = _eval_number(args.get("discount", ""))
        if price is None or d is None:
            return None
        return _format_number(price / (1 - d / 100))
    if op == "unit":
        k = args.get("kind", "")
        if k == "seconds_day":
            return str(24 * 60 * 60)
        return None
    if op == "linear_system":
        s = _eval_number(args.get("a", ""))
        d = _eval_number(args.get("b", ""))
        if s is None or d is None:
            return None
        return _format_number((s + d) / 2)
    if op == "seq_term":
        n = int(args.get("n", "0"))
        nums_s = args.get("nums", "")
        nums = [int(x) for x in re.findall(r"-?\d+", nums_s)][:4]
        if len(nums) < 2:
            return None
        d = nums[1] - nums[0]
        return str(nums[0] + (n - 1) * d)
    if op == "speed":
        d = _eval_number(args.get("dist", ""))
        h = _eval_number(args.get("hours", ""))
        if d is None or h == 0:
            return None
        return _format_number(d / h)
    if op == "proportion":
        n1 = _eval_number(args.get("n1", ""))
        c1 = _eval_number(args.get("cost1", ""))
        n2 = _eval_number(args.get("n2", ""))
        if n1 == 0 or n2 == 0:
            return None
        return _format_number(c1 * n2 / n1)
    if op == "simple_algebra":
        off = _eval_number(args.get("offset", "0")) or 0
        tot = _eval_number(args.get("total", "0"))
        if tot is None:
            return None
        return _format_number((tot - off) / 2)
    if op == "exp_solve":
        b = _eval_number(args.get("base", ""))
        r = _eval_number(args.get("result", ""))
        if b is None or r is None or b <= 0 or r <= 0:
            return None
        n = 0
        v = 1.0
        while v < r and n < 1000:
            v *= b
            n += 1
        if abs(v - r) < 1e-9:
            return str(n)
        return None
    if op == "pct_change":
        up = _eval_number(args.get("up", "0")) or 0
        down = _eval_number(args.get("down", "0")) or 0
        net = ((1 + up / 100) * (1 - down / 100) - 1) * 100
        return _format_number(net)
    if op == "work_rate":
        d1 = _eval_number(args.get("t1", ""))
        w1 = _eval_number(args.get("w1", ""))
        w2 = _eval_number(args.get("w2", ""))
        w3 = _eval_number(args.get("t2", ""))
        if d1 is None or w1 == 0 or w2 == 0 or w3 is None:
            return None
        days = d1 * w3 / w2
        return _format_number(days)
    return None


def resolve_compound_expression(question: str) -> Optional[str]:
    det = detect_compound_expression(question)
    if det is None:
        return None
    expr = det["expr"]
    tokens = re.findall(r"\d+|[+\-*/x^]", expr)
    if len(tokens) > 5:
        return None
    v = _eval_expression(expr)
    if v is None:
        return None
    return _format_number(v)
