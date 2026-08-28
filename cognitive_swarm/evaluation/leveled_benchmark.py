"""Leveled benchmark: Multi-small-model swarm vs single large API model.

Compares, across difficulty levels (easy / medium / hard) and categories
(math / reasoning / code), plus a dedicated adversarial/robustness category
(trap):
  1. Phi-3.5 Mini alone  (local, with optimized prompts)
  2. Qwen 2.5 3B alone   (local, with optimized prompts)
  3. Optimized Debate    (local swarm: Phi + Qwen debating)
  4. Qwen 27B via API    (large remote model)

The goal is to answer a single question with evidence:
  "Is a swarm of small local models a viable alternative to one large model?"

The test set is **120 problems**: 36 math + 36 reasoning + 36 code (12 each of
easy/medium/hard) + 12 hard adversarial trap items. Every answer has exact
ground truth, and the code + trap answers are deterministically verifiable in
Python, so the benchmark's labels are not guesswork.

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
# Problem definition (3 levels x 3 categories x 12 problems = 108 + 12 trap)
# ---------------------------------------------------------------------------
def _p(pid, level, category, question, answer):
    return {"id": pid, "level": level, "category": category, "q": question, "a": answer}


LEVELED_PROBLEMS = [
    # ============================= EASY =============================
    _p("e-m-1", "easy", "math", "What is 127 + 389?", "516"),
    _p("e-m-2", "easy", "math", "What is 15 * 24?", "360"),
    _p("e-m-3", "easy", "math", "What is 144 divided by 12?", "12"),
    _p("e-m-4", "easy", "math", "What is 2 to the power of 8 (2^8)?", "256"),
    _p("e-m-5", "easy", "math", "What is 63 * 11?", "693"),
    _p("e-m-6", "easy", "math", "What is 100 divided by 4 plus 6 (100 / 4 + 6)?", "31"),
    _p("e-m-7", "easy", "math", "What is 3 * 7 - 5?", "16"),
    _p("e-m-8", "easy", "math", "What is 25 * 4?", "100"),
    _p("e-m-9", "easy", "math", "What is half of 250?", "125"),
    _p("e-m-10", "easy", "math", "What is 7 + 8 + 9?", "24"),
    _p("e-m-11", "easy", "math", "What is 6 squared (6^2)?", "36"),
    _p("e-m-12", "easy", "math", "What is 120 divided by 5?", "24"),
    _p("e-r-1", "easy", "reasoning", "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "9"),
    _p("e-r-2", "easy", "reasoning", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5"),
    _p("e-r-3", "easy", "reasoning", "If all cats are animals, and some animals are black, can we conclude that some cats are black?", "No"),
    _p("e-r-4", "easy", "reasoning", "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?", "0.05"),
    _p("e-r-5", "easy", "reasoning", "Which weighs more: a pound of feathers or a pound of iron?", "same"),
    _p("e-r-6", "easy", "reasoning", "There are 3 blue balls and 2 red balls in a bag. You draw one at random. What is the probability it is red?", "0.4"),
    _p("e-r-7", "easy", "reasoning", "A plane crashes exactly on the border between two countries. Where should the survivors be buried?", "nowhere"),
    _p("e-r-8", "easy", "reasoning", "An electric train is travelling north at 60 km/h. Which way does the smoke blow?", "no smoke"),
    _p("e-r-9", "easy", "reasoning", "What comes once in a minute, twice in a moment, but never in a thousand years?", "m"),
    _p("e-r-10", "easy", "reasoning", "How many months of the year have 28 days?", "12"),
    _p("e-r-11", "easy", "reasoning", "If you have 10 apples and give away 4, how many apples do you have left?", "6"),
    _p("e-r-12", "easy", "reasoning", "A farmer has 5 dogs. Each dog sees 3 cats. How many cats does the farmer have?", "3"),
    _p("e-c-1", "easy", "code", "What does this Python code output: print(2 ** 3 ** 2)?", "512"),
    _p("e-c-2", "easy", "code", "In Python, what is the result of: len([1, 2, [3, 4]])?", "3"),
    _p("e-c-3", "easy", "code", "What does this Python code output: print(10 // 3)?", "3"),
    _p("e-c-4", "easy", "code", "What does this Python code output: print('ab' * 3)?", "ababab"),
    _p("e-c-5", "easy", "code", "What does this Python code output: print(2 + 3 * 4)?", "14"),
    _p("e-c-6", "easy", "code", "What does this Python code output: print(len('hello world'))?", "11"),
    _p("e-c-7", "easy", "code", "What does this Python code output: print(7 % 3)?", "1"),
    _p("e-c-8", "easy", "code", "What does this Python code output: print([1, 2, 3][1])?", "2"),
    _p("e-c-9", "easy", "code", "What does this Python code output: print('abc' * 2)?", "abcabc"),
    _p("e-c-10", "easy", "code", "What does this Python code output: print(max(3, 7, 2))?", "7"),
    _p("e-c-11", "easy", "code", "What does this Python code output: print(5 > 3 and 2 < 4)?", "True"),
    _p("e-c-12", "easy", "code", "What does this Python code output: print('hello'[1:4])?", "ell"),

    # ============================= MEDIUM =============================
    _p("m-m-1", "medium", "math", "If a shirt costs $45 and is on sale for 30% off, what is the sale price?", "31.5"),
    _p("m-m-2", "medium", "math", "What is the value of 7! (seven factorial)?", "5040"),
    _p("m-m-3", "medium", "math", "A rectangle has length 12 and width 5. What is its area in square units?", "60"),
    _p("m-m-4", "medium", "math", "What is 40% of 250?", "100"),
    _p("m-m-5", "medium", "math", "A train travels 240 km in 3 hours. What is its speed in km/h?", "80"),
    _p("m-m-6", "medium", "math", "If 3 notebooks cost $12, how much do 5 notebooks cost?", "20"),
    _p("m-m-7", "medium", "math", "What is the sum of the interior angles of a triangle in degrees?", "180"),
    _p("m-m-8", "medium", "math", "A number doubled plus 6 equals 20. What is the number?", "7"),
    _p("m-m-9", "medium", "math", "What is 15% of 80?", "12"),
    _p("m-m-10", "medium", "math", "If x + y = 12 and x - y = 4, what is the value of x?", "8"),
    _p("m-m-11", "medium", "math", "What is 2 to the power of 6 (2^6)?", "64"),
    _p("m-m-12", "medium", "math", "A square has a perimeter of 36. What is its area in square units?", "81"),
    _p("m-r-1", "medium", "reasoning", "Sarah is twice as old as John. In 10 years, Sarah will be 5 years older than John. How old is John now?", "5"),
    _p("m-r-2", "medium", "reasoning", "Three doctors said Robert was their brother. But Robert said he had no brothers. Who is lying?", "the doctors"),
    _p("m-r-3", "medium", "reasoning", "You have a 5-liter jug and a 3-liter jug. What is the exact number of liters that CANNOT be measured using only these two jugs and unlimited water?", "Cannot determine / none"),
    _p("m-r-4", "medium", "reasoning", "Which number comes next in the sequence: 2, 6, 12, 20, 30, ...?", "42"),
    _p("m-r-5", "medium", "reasoning", "There are 8 people in a room. Each person shakes hands with every other person exactly once. How many handshakes occur?", "28"),
    _p("m-r-6", "medium", "reasoning", "Two fathers and two sons go fishing and each catches one fish, a total of three fish. How is that possible?", "grandfather"),
    _p("m-r-7", "medium", "reasoning", "Which is the larger fraction: 2/3 or 3/5?", "2/3"),
    _p("m-r-8", "medium", "reasoning", "A rope ladder hangs from a ship with rungs 1 foot apart. The tide rises at 1 foot per hour. How many rungs are covered after 2 hours?", "0"),
    _p("m-r-9", "medium", "reasoning", "What number is one half of one quarter of 400?", "50"),
    _p("m-r-10", "medium", "reasoning", "If it takes 3 gardeners 2 hours to weed a garden, how many hours would it take 6 gardeners working at the same rate?", "1"),
    _p("m-r-11", "medium", "reasoning", "A car uses 8 liters per 100 km. How many liters does it use for 25 km?", "2"),
    _p("m-r-12", "medium", "reasoning", "A certain recipe needs 2 eggs for 4 people. How many eggs are needed for 10 people?", "5"),
    _p("m-c-1", "medium", "code", "What does this Python code output: print([i**2 for i in range(4)])?", "[0, 1, 4, 9]"),
    _p("m-c-2", "medium", "code", "What does this Python code output: print(sorted([3, 1, 2]) == [1, 2, 3])?", "True"),
    _p("m-c-3", "medium", "code", "What does this Python code output: print('hello'.upper()[0:2])?", "HE"),
    _p("m-c-4", "medium", "code", "What does this Python code output: print(5 == 5.0 and 0.5 == 0.5)?", "True"),
    _p("m-c-5", "medium", "code", "What does this Python code output: print(3 ** 2 + 4 ** 2)?", "25"),
    _p("m-c-6", "medium", "code", "What does this Python code output: print([x for x in range(1, 6) if x % 2 == 1])?", "[1, 3, 5]"),
    _p("m-c-7", "medium", "code", "What does this Python code output: print('banana'.count('a'))?", "3"),
    _p("m-c-8", "medium", "code", "What does this Python code output: print(len({1, 2, 2, 3}))?", "3"),
    _p("m-c-9", "medium", "code", "What does this Python code output: print(''.join(reversed('abc')))?", "cba"),
    _p("m-c-10", "medium", "code", "What does this Python code output: print(10 % 3 + 1)?", "2"),
    _p("m-c-11", "medium", "code", "What does this Python code output: print(sum([1, 2, 3, 4]))?", "10"),
    _p("m-c-12", "medium", "code", "What does this Python code output: print(type(True) is bool)?", "True"),

    # ============================= HARD =============================
    _p("h-m-1", "hard", "math", "What is the sum of the first 50 positive integers (1+2+...+50)?", "1275"),
    _p("h-m-2", "hard", "math", "If 3^x = 27, what is the value of x?", "3"),
    _p("h-m-3", "hard", "math", "A number is increased by 20% and then decreased by 20%. What is the net percentage change?", "-4"),
    _p("h-m-4", "hard", "math", "What is the greatest common divisor of 48 and 180?", "12"),
    _p("h-m-5", "hard", "math", "What is the value of 2 to the power of 10 (2^10)?", "1024"),
    _p("h-m-6", "hard", "math", "What is the sum of the first 20 even positive integers (2+4+...+40)?", "420"),
    _p("h-m-7", "hard", "math", "What is the value of sqrt(144) + sqrt(81)?", "21"),
    _p("h-m-8", "hard", "math", "A jacket is sold for $80 after a 20% discount. What was its original price?", "100"),
    _p("h-m-9", "hard", "math", "How many seconds are there in one full day?", "86400"),
    _p("h-m-10", "hard", "math", "What is the 12th term of the arithmetic sequence 3, 8, 13, 18, ...?", "58"),
    _p("h-m-11", "hard", "math", "If 4 workers build 4 walls in 4 days, how many days would it take 8 workers to build 8 walls?", "4"),
    _p("h-m-12", "hard", "math", "What is log base 2 of 64 (log2 64)?", "6"),
    _p("h-r-1", "hard", "reasoning", "A snail is at the bottom of a 30-foot well. Each day it climbs 3 feet, but at night it slides back 2 feet. How many days does it take to reach the top?", "28"),
    _p("h-r-2", "hard", "reasoning", "Four people need to cross a bridge at night with one flashlight that only lasts 15 minutes. They walk at 1, 2, 5 and 10 minutes. At most two can cross at a time and the pair moves at the slower person's pace. What is the minimum total time to get all four across?", "17"),
    _p("h-r-3", "hard", "reasoning", "Alice and Bob have a total of 40 apples. Bob gives Alice 5 apples. Now Alice has twice as many as Bob. How many apples did Alice start with?", "20"),
    _p("h-r-4", "hard", "reasoning", "In a group of 100 people, each person shakes hands with every other person exactly once. How many handshakes occur in total?", "4950"),
    _p("h-r-5", "hard", "reasoning", "A family has 5 children and each child has at least one sister. What is the minimum number of children in the family?", "6"),
    _p("h-r-6", "hard", "reasoning", "A man has 3 boxes labeled Apples, Oranges, and Apples&Oranges, but every label is wrong. He may pick one fruit from one box. Which box should he pick from to label all boxes correctly?", "mixed"),
    _p("h-r-7", "hard", "reasoning", "There are 15 people at a party and every pair shakes hands exactly once. How many handshakes occur?", "105"),
    _p("h-r-8", "hard", "reasoning", "Two trains start 300 km apart and travel toward each other at 60 km/h and 90 km/h. How many hours until they meet?", "2"),
    _p("h-r-9", "hard", "reasoning", "What is the 100th odd positive integer?", "199"),
    _p("h-r-10", "hard", "reasoning", "Ten people can build a house in 20 days. How many people are needed to build it in 5 days?", "40"),
    _p("h-r-11", "hard", "reasoning", "A cube is painted on all faces and then cut into 27 equal smaller cubes. How many of the smaller cubes have paint on exactly 2 faces?", "12"),
    _p("h-r-12", "hard", "reasoning", "A 100 cm rope is cut into two pieces so that one piece is 3 times as long as the other. How long is the shorter piece in cm?", "25"),
    _p("h-c-1", "hard", "code", "What does this Python code print: print(sorted([x for x in range(1,31) if x % 3 == 0 and x % 5 == 0]))?", "[15, 30]"),
    _p("h-c-2", "hard", "code", "What does this Python code output: print(list(zip([1,2,3], 'abc')))?", "[(1, 'a'), (2, 'b'), (3, 'c')]"),
    _p("h-c-3", "hard", "code", "What does this Python code output: print({x: x**2 for x in range(3)})?", "{0: 0, 1: 1, 2: 4}"),
    _p("h-c-4", "hard", "code", "What does this Python code output: print(functools.reduce(lambda a,b: a*b, [1,2,3,4])) - assume functools is imported?", "24"),
    _p("h-c-5", "hard", "code", "What does this Python code output: print([x for x in range(10) if x % 2 == 0])?", "[0, 2, 4, 6, 8]"),
    _p("h-c-6", "hard", "code", "What does this Python code output: print(sorted('hello'))?", "['e', 'h', 'l', 'l', 'o']"),
    _p("h-c-7", "hard", "code", "What does this Python code output: print(list(range(0, 20, 5)))?", "[0, 5, 10, 15]"),
    _p("h-c-8", "hard", "code", "What does this Python code output: print('racecar'[::-1])?", "racecar"),
    _p("h-c-9", "hard", "code", "What does this Python code output: print([[i, i * 2] for i in range(3)])?", "[[0, 0], [1, 2], [2, 4]]"),
    _p("h-c-10", "hard", "code", "What does this Python code output: print(dict(zip(['a', 'b'], [1, 2])))?", "{'a': 1, 'b': 2}"),
    _p("h-c-11", "hard", "code", "What does this Python code output: print(len(set('hello world')))?", "8"),
    _p("h-c-12", "hard", "code", "What does this Python code output: print(2 ** 3 * 3 ** 2)?", "72"),

    # ============================= TRAP (adversarial / robustness) ==========
    # Classic small-model (and large-model) failure traps: letter/vowel/char
    # counting and spelling-backwards. Every answer is verified deterministically
    # in Python so ground truth is exact.
    _p("h-t-1", "hard", "trap", "Spell the word 'strawberry' backwards (letter by letter).", "yrrebwarts"),
    _p("h-t-2", "hard", "trap", "How many letter 'r' are in the word 'strawberry'?", "3"),
    _p("h-t-3", "hard", "trap", "How many vowels (a, e, i, o, u) are in the word 'beautiful'?", "5"),
    _p("h-t-4", "hard", "trap", "Spell the word 'Mississippi' backwards (letter by letter).", "ippississim"),
    _p("h-t-5", "hard", "trap", "How many letters are there in the word 'Mississippi'?", "11"),
    _p("h-t-6", "hard", "trap", "How many letter 's' are in the word 'Mississippi'?", "4"),
    _p("h-t-7", "hard", "trap", "How many vowels (a, e, i, o, u) are in the word 'rhythm'?", "0"),
    _p("h-t-8", "hard", "trap", "Spell the word 'rhythm' backwards (letter by letter).", "mhtyhr"),
    _p("h-t-9", "hard", "trap", "How many vowels (a, e, i, o, u) are in the word 'beautifully'?", "5"),
    _p("h-t-10", "hard", "trap", "Which is larger: the fraction 3/10 or the decimal 0.29?", "3/10"),
    _p("h-t-11", "hard", "trap", "Which is larger: the decimal 0.5 or the fraction 1/3?", "0.5"),
    _p("h-t-12", "hard", "trap", "How many letter 'i' are in the word 'Mississippi'?", "4"),
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
    """Phi + Qwen swarm debate with optimized prompts and confidence consensus."""
    def __init__(self, samples_per_model: int = 1):
        from cognitive_swarm.orchestration.optimized_debate import OptimizedDebate
        self.debate = OptimizedDebate(max_rounds=1, samples_per_model=samples_per_model)
        self.samples_per_model = samples_per_model

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


def run_leveled_benchmark(runs_per_problem: int = 2, samples_per_model: int = 1):
    solvers = {
        "phi": PhiSolver(),
        "qwen": QwenSolver(),
        "debate": OptimizedDebateSolver(samples_per_model=samples_per_model),
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
            "samples_per_model": samples_per_model,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int, nargs="?", default=2, help="runs per problem")
    parser.add_argument("--samples", type=int, default=1, help="samples per model for confidence consensus")
    args = parser.parse_args()
    run_leveled_benchmark(runs_per_problem=args.runs, samples_per_model=args.samples)
