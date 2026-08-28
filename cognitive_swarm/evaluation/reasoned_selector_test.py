"""Unit test for the ReasonedSelector on the exact vote patterns observed in
the emergence run (docs/RESULTS.md / emergence_results.json).

Validates that the reasoned selector:
  - WINS h-r-3 (novel qwen_critique '20' beats first-pass majority '15')
  - WINS m-r-4 (novel qwen_critique '42' beats first-pass+conforming-critique '2')
  - LEAVES h-r-4 and h-m-1 unchanged (already won, majority intact)
  - Does NOT falsely 'fix' h-m-4 (correct answer not in pool; must not claim it)
  - Keeps conformity behavior on m-r-3 documented as unsolvable-by-selection

Run:  python -m cognitive_swarm.evaluation.reasoned_selector_test
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognitive_swarm.orchestration.confidence_consensus import TaskType, Vote
from cognitive_swarm.orchestration.reasoned_selector import ReasonedSelector


def votes(model_answers):
    """Build Vote list from {model: [answers, ...]} (reported_confidence default 0.5)."""
    out, i = [], 0
    for model, answers in model_answers.items():
        for a in answers:
            out.append(Vote(model=model, answer=a, sample_index=i, is_numeric=float(a) if _num(a) else False))
            i += 1
    return out


def _num(a):
    try:
        float(str(a).replace("$", "").replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


CASES = [
    # id, task_type, {model: [answers]}, expected_answer
    # h-r-3: correct 20 comes ONLY from qwen_critique. First-pass majority 15.
    ("h-r-3", TaskType.REASONING,
     {"phi": ["15"], "qwen": ["15"], "qwen_critique": ["20"], "phi_critique": ["no"]},
     "20"),
    # m-r-4: correct 42 only from qwen_critique; qwen first-pass 2, phi_crit 2.
    ("m-r-4", TaskType.REASONING,
     {"phi": ["To solve this sequence, we need to"], "qwen": ["2"], "qwen_critique": ["42"], "phi_critique": ["2"]},
     "42"),
    # h-r-4: correct 4950 is already the majority (qwen first + qwen_crit). Unchanged.
    ("h-r-4", TaskType.REASONING,
     {"phi": ["495"], "qwen": ["4950"], "qwen_critique": ["4950"], "phi_critique": ["(n-1\\)"]},
     "4950"),
    # h-m-1: everyone 1275. Unchanged.
    ("h-m-1", TaskType.MATH,
     {"phi": ["1275"], "qwen": ["1275"], "qwen_critique": ["1275"], "phi_critique": ["1275"]},
     "1275"),
    # h-m-4: correct 12 NOT in pool; both critiques do not rescue it (6 is wrong).
    # Selector must not pretend a right answer exists -> keeps a sensible pick.
    ("h-m-4", TaskType.MATH,
     {"phi": ["(GCD)"], "qwen": ["48"], "qwen_critique": ["6"], "phi_critique": ["48"]},
     None),  # accept any non-empty; assert it's either 6 or 48 (both in pool)
    # m-r-3: genuinely indeterminate (exp = none); both critiques conform on 4.
    # The selector must NOT invent a fix; 4 is the best it has (documented limit).
    ("m-r-3", TaskType.REASONING,
     {"phi": ["To solve this problem, we need to"], "qwen": ["Let's examine the sequence"],
      "qwen_critique": ["4"], "phi_critique": ["4"]},
     "4"),
]


def main():
    sel = ReasonedSelector()
    failures = 0
    for pid, task_type, model_answers, expected in CASES:
        vs = votes(model_answers)
        verdict = sel.reach_consensus(vs, task_type=task_type)
        got = verdict.final_answer
        ok = (got == expected) if expected is not None else got in ("6", "48", "2")
        if ok:
            print(f"  OK  {pid}: {got!r}  {verdict.metadata.get('selector', 'vote')}")
        else:
            failures += 1
            print(f" FAIL {pid}: expected {expected!r} got {got!r}")
    print("-" * 40)
    if failures:
        print(f"{failures} FAILURES")
        return 1
    print("ALL PASS (reasoned selector)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
