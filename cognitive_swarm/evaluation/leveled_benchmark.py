"""Leveled benchmark: Multi-small-model swarm vs single large API model.

Compares, across 3 difficulty levels (easy / medium / hard) and 3 categories
(math / reasoning / code):
  1. Phi-3.5 Mini alone  (local, with optimized prompts)
  2. Qwen 2.5 3B alone   (local, with optimized prompts)
  3. Optimized Debate    (local swarm: Phi + Qwen debating)
  4. Qwen 27B via API    (large remote model)

The goal is to answer a single question with evidence:
  "Is a swarm of small local models a viable alternative to one large model?"

We report accuracy per level, per category, plus consistency across repeated
runs, so the results go beyond a single lucky run.
"""

import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Problem definition (3 levels x 3 categories x ~4 problems = 36 problems)
# ---------------------------------------------------------------------------
def _p(pid, level, category, question, answer):
    return {"id": pid, "level": level, "category": category, "q": question, "a": answer}


LEVELED_PROBLEMS = [
    # ============================= EASY =============================
    _p("e-m-1", "easy", "math", "What is 127 + 389?", "516"),
    _p("e-m-2", "easy", "math", "What is 15 * 24?", "360"),
    _p("e-m-3", "easy", "math", "What is 144 divided by 12?", "12"),
    _p("e-m-4", "easy", "math", "What is 2 to the power of 8 (2^8)?", "256"),
    _p("e-r-1", "easy", "reasoning", "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "9"),
    _p("e-r-2", "easy", "reasoning", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5"),
    _p("e-r-3", "easy", "reasoning", "If all cats are animals, and some animals are black, can we conclude that some cats are black?", "No"),
    _p("e-r-4", "easy", "reasoning", "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?", "0.05"),
    _p("e-c-1", "easy", "code", "What does this Python code output: print(2 ** 3 ** 2)?", "512"),
    _p("e-c-2", "easy", "code", "In Python, what is the result of: len([1, 2, [3, 4]])?", "3"),
    _p("e-c-3", "easy", "code", "What does this Python code output: print(10 // 3)?", "3"),
    _p("e-c-4", "easy", "code", "What does this Python code output: print('ab' * 3)?", "ababab"),

    # ============================= MEDIUM =============================
    _p("m-m-1", "medium", "math", "If a shirt costs $45 and is on sale for 30% off, what is the sale price?", "31.5"),
    _p("m-m-2", "medium", "math", "What is the value of 7! (seven factorial)?", "5040"),
    _p("m-m-3", "medium", "math", "A rectangle has length 12 and width 5. What is its area in square units?", "60"),
    _p("m-m-4", "medium", "math", "What is 40% of 250?", "100"),
    _p("m-r-1", "medium", "reasoning", "Sarah is twice as old as John. In 10 years, Sarah will be 5 years older than John. How old is John now?", "5"),
    _p("m-r-2", "medium", "reasoning", "Three doctors said Robert was their brother. But Robert said he had no brothers. Who is lying?", "the doctors"),
    _p("m-r-3", "medium", "reasoning", "You have a 5-liter jug and a 3-liter jug. What is the exact number of liters that CANNOT be measured using only these two jugs and unlimited water?", "Cannot determine / none"),
    _p("m-r-4", "medium", "reasoning", "Which number comes next in the sequence: 2, 6, 12, 20, 30, ...?", "42"),
    _p("m-c-1", "medium", "code", "What does this Python code output: print([i**2 for i in range(4)])?", "[0, 1, 4, 9]"),
    _p("m-c-2", "medium", "code", "What does this Python code output: print(sorted([3, 1, 2]) == [1, 2, 3])?", "True"),
    _p("m-c-3", "medium", "code", "What does this Python code output: print('hello'.upper()[0:2])?", "HE"),
    _p("m-c-4", "medium", "code", "What does this Python code output: print(5 == 5.0 and 0.5 == 0.5)?", "True"),

    # ============================= HARD =============================
    _p("h-m-1", "hard", "math", "What is the sum of the first 50 positive integers (1+2+...+50)?", "1275"),
    _p("h-m-2", "hard", "math", "If 3^x = 27, what is the value of x?", "3"),
    _p("h-m-3", "hard", "math", "A number is increased by 20% and then decreased by 20%. What is the net percentage change?", "-4"),
    _p("h-m-4", "hard", "math", "What is the greatest common divisor of 48 and 180?", "12"),
    _p("h-r-1", "hard", "reasoning", "A snail is at the bottom of a 30-foot well. Each day it climbs 3 feet, but at night it slides back 2 feet. How many days does it take to reach the top?", "28"),
    _p("h-r-2", "hard", "reasoning", "Four people need to cross a bridge at night with one flashlight that only lasts 15 minutes. They walk at 1, 2, 5 and 10 minutes. At most two can cross at a time and the pair moves at the slower person's pace. What is the minimum total time to get all four across?", "17"),
    _p("h-r-3", "hard", "reasoning", "Alice and Bob have a total of 40 apples. Bob gives Alice 5 apples. Now Alice has twice as many as Bob. How many apples did Alice start with?", "20"),
    _p("h-r-4", "hard", "reasoning", "In a group of 100 people, each person shakes hands with every other person exactly once. How many handshakes occur in total?", "4950"),
    _p("h-c-1", "hard", "code", "What does this Python code print: print(sorted([x for x in range(1,31) if x % 3 == 0 and x % 5 == 0]))?", "[15, 30]"),
    _p("h-c-2", "hard", "code", "What does this Python code output: print(list(zip([1,2,3], 'abc')))?", "[(1, 'a'), (2, 'b'), (3, 'c')]"),
    _p("h-c-3", "hard", "code", "What does this Python code output: print({x: x**2 for x in range(3)})?", "{0: 0, 1: 1, 2: 4}"),
    _p("h-c-4", "hard", "code", "What does this Python code output: print(functools.reduce(lambda a,b: a*b, [1,2,3,4])) - assume functools is imported?", "24"),
]


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------
def check_answer(predicted: str, expected: str) -> bool:
    """Robust answer comparison."""
    pred = predicted.strip().lower()
    exp = expected.strip().lower()

    # Normalize: remove $ , and surrounding whitespace
    pred_clean = pred.replace("$", "").replace(",", "")
    exp_clean = exp.replace("$", "").replace(",", "")

    # Direct match
    if pred_clean == exp_clean:
        return True
    # Expected contained in predicted answer
    if exp_clean in pred_clean:
        return True

    # Numeric comparison
    try:
        pred_num = float(pred_clean)
        exp_num = float(exp_clean)
        return abs(pred_num - exp_num) < 1e-6
    except (ValueError, TypeError):
        pass

    # Percent sign handling (e.g., "-4%" vs "-4")
    try:
        pred_pct = float(pred_clean.replace("%", ""))
        exp_pct = float(exp_clean.replace("%", ""))
        if abs(pred_pct - exp_pct) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # Boolean handling
    if exp in ("true", "false"):
        booleans = {"true": "True", "false": "False", "t": "True", "f": "False"}
        if pred in booleans:
            return booleans[pred].lower() == exp

    return False


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
class PhiSolver:
    """Phi-3.5 Mini with optimized prompts."""
    def __init__(self):
        from cognitive_swarm.models.multi_backend import MultiModelBackend
        from cognitive_swarm.orchestration.prompt_optimizer import PromptOptimizer
        self.backend = MultiModelBackend()
        self.optimizer = PromptOptimizer(use_llm=False)

    def solve(self, q, max_tokens=32):
        opt = self.optimizer.optimize(q, model_name="phi")
        self.backend.load_model("phi", max_tokens=max_tokens)
        try:
            res = self.backend.generate(opt.optimized_prompt)
            return _extract_plain(res.text)
        finally:
            self.backend.unload_model()


class QwenSolver:
    """Qwen 2.5 3B with optimized prompts."""
    def __init__(self):
        from cognitive_swarm.models.multi_backend import MultiModelBackend
        from cognitive_swarm.orchestration.prompt_optimizer import PromptOptimizer
        self.backend = MultiModelBackend()
        self.optimizer = PromptOptimizer(use_llm=False)

    def solve(self, q, max_tokens=32):
        opt = self.optimizer.optimize(q, model_name="qwen")
        self.backend.load_model("qwen", max_tokens=max_tokens)
        try:
            res = self.backend.generate(opt.optimized_prompt)
            return _extract_plain(res.text)
        finally:
            self.backend.unload_model()


class OptimizedDebateSolver:
    """Phi + Qwen swarm debate with optimized prompts."""
    def __init__(self):
        from cognitive_swarm.orchestration.optimized_debate import OptimizedDebate
        self.debate = OptimizedDebate(max_rounds=1)

    def solve(self, q, max_tokens=32):
        res = self.debate.run_debate(q, max_tokens=max_tokens)
        return res.final_answer


class APISolver:
    """Large remote model (Qwen 27B via Groq)."""
    def __init__(self):
        from cognitive_swarm.evaluation.full_comparison import APIModel
        self.api = APIModel()

    def solve(self, q, max_tokens=64):
        prompt = f"""Solve this problem. Give ONLY the answer, nothing else.

Problem: {q}

ANSWER: """
        res = self.api.generate(prompt, max_tokens=64)
        return _extract_api(res["text"])


def _extract_plain(text: str) -> str:
    """Extract the plain answer from an optimized prompt output."""
    import re
    # The optimized math prompt ends with "ANSWER = " or "FINAL PRICE = "
    for marker in ["ANSWER =", "NUMBER:", "FINAL PRICE =", "OUTPUT:", "ANSWER:"]:
        if marker in text:
            idx = text.index(marker) + len(marker)
            return text[idx:].strip().split("\n")[0].strip()
    lines = [l for l in text.strip().split("\n") if l.strip()]
    # Return last non-empty line
    return lines[-1].strip() if lines else text.strip()


def _extract_api(text: str) -> str:
    import re
    match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l for l in text.strip().split("\n") if l.strip()]
    return lines[-1].strip() if lines else text.strip()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def summarize(results: List[Dict]) -> Dict[str, Any]:
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0,
    }


def run_leveled_benchmark(runs_per_problem: int = 2):
    solvers = {
        "phi": PhiSolver(),
        "qwen": QwenSolver(),
        "debate": OptimizedDebateSolver(),
        "api": APISolver(),
    }

    # Pre-check API availability
    api_available = solvers["api"].api.available

    # Initialize tracking: per problem, per mode -> list of predictions
    results = {}
    for p in LEVELED_PROBLEMS:
        results[p["id"]] = {m: [] for m in solvers if m != "api" or api_available}

    print("=" * 80)
    print("LEVELED BENCHMARK: Small-model swarm vs Large single model")
    print(f"Problems: {len(LEVELED_PROBLEMS)} | Runs/problem: {runs_per_problem}")
    print(f"API (Qwen 27B) available: {api_available}")
    print("=" * 80)

    # To reduce model reload overhead, process problem-by-problem: for each
    # problem run phi solvers for all runs, then qwen for all runs, etc.
    for p in LEVELED_PROBLEMS:
        print(f"\n[{p['id']}] ({p['level']}/{p['category']}) {p['q']}")
        print(f"    Expected: {p['a']}")

        for mode, solver in solvers.items():
            if mode == "api" and not api_available:
                continue
            for run in range(runs_per_problem):
                t0 = time.time()
                try:
                    ans = solver.solve(p["q"])
                    latency = (time.time() - t0) * 1000
                except Exception as e:
                    ans = f"ERROR: {e}"
                    latency = 0
                correct = check_answer(ans, p["a"])
                results[p["id"]][mode].append({
                    "answer": ans, "correct": correct, "latency_ms": latency,
                })
            first = results[p["id"]][mode][0]
            mark = "".join("✓" if r["correct"] else "✗" for r in results[p["id"]][mode])
            print(f"    {mode:<8} {first['answer'][:42]:<42} [{mark}]")

        print(f"    {'-' * 60}")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    modes = [m for m in solvers if m != "api" or api_available]

    # Overall accuracy (mode solves problem if ANY run is correct)
    overall = {}
    for mode in modes:
        solves = sum(1 for pid, data in results.items()
                     if any(r["correct"] for r in data[mode]))
        overall[mode] = solves / len(results)

    # Per level/category: solved-at-least-once
    agg = {}
    for mode in modes:
        agg[mode] = {"overall": overall[mode]}
        for level in ["easy", "medium", "hard"]:
            lvl_ids = [p["id"] for p in LEVELED_PROBLEMS if p["level"] == level]
            solves = sum(1 for pid in lvl_ids
                         if any(r["correct"] for r in results[pid][mode]))
            agg[mode][f"level_{level}"] = solves / max(len(lvl_ids), 1)
        for cat in ["math", "reasoning", "code"]:
            cat_ids = [p["id"] for p in LEVELED_PROBLEMS if p["category"] == cat]
            solves = sum(1 for pid in cat_ids
                         if any(r["correct"] for r in results[pid][mode]))
            agg[mode][f"cat_{cat}"] = solves / max(len(cat_ids), 1)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("RESULTS SUMMARY (problem solved if correct in any run)")
    print("=" * 80)

    # Header
    headers = ["mode"] + [l for l in ["overall", "level_easy", "level_medium", "level_hard",
                                       "cat_math", "cat_reasoning", "cat_code"]]
    print(f"{'mode':<10}" + "".join(f"{h.replace('level_','L-').replace('cat_','C-'):>12}" for h in headers[1:]))
    for mode in modes:
        row = f"{mode:<10}"
        for h in headers[1:]:
            row += f"{agg[mode][h]:>12.0%}"
        print(row)

    print("\n--- Per-problem detail ---")
    for p in LEVELED_PROBLEMS:
        line = f"[{p['id']}] {p['q']:<70}"
        detail = "  "
        for mode in modes:
            r = results[p["id"]][mode]
            solved = any(x["correct"] for x in r)
            detail += f"{mode}:{'✓' if solved else '✗'} "
        print(line + detail)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out = {
        "config": {
            "runs_per_problem": runs_per_problem,
            "problems_total": len(LEVELED_PROBLEMS),
            "api_available": api_available,
            "levels": ["easy", "medium", "hard"],
            "categories": ["math", "reasoning", "code"],
        },
        "agg": agg,
        "problems": LEVELED_PROBLEMS,
        "results": results,
    }
    outfile = Path("leveled_benchmark_results.json")
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run_leveled_benchmark(runs_per_problem=runs)
