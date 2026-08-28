"""Big-model comparator (OpenRouter) against the same 120-problem benchmark.

Answer to the credibility gap: to claim a local small-model swarm rivals a
*large* model, we must actually score a large model on the SAME problems with
the SAME ground-truth scorer. This runs a large model (default
`minimax/minimax-m3:free` via OpenRouter, a frontier-class free model that
honors a terse output contract) on LEVELED_PROBLEMS and reports mean accuracy
per level/category + bootstrap 95% CI, identical to `validate_baseline.py` so
the numbers are apples-to-apples.

Fairness: the big model gets the same tight answer-first OUTPUT CONTRACT as the
local models (per-category prompt ending in an `ANSWER:` marker, small token
budget, told not to output chain-of-thought). This deliberately avoids the old
v1/v2 artifact where a chatty "thinking-mode" big model was regex-mangled into a
wrong answer. The big model is asked for a clean answer, like everyone else.

Token is read from OPENROUTER_API_KEY (or --key). Never hardcoded/committed.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python -m cognitive_swarm.evaluation.big_model_benchmark --model openai/gpt-4o \
      --subset balanced --out results_big_balanced.json
  python -m cognitive_swarm.evaluation.big_model_benchmark --model openai/gpt-4o \
      --subset all --max-tokens 40 --out results_big_all.json
"""
import sys, os, json, time, argparse, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognitive_swarm.evaluation.leveled_benchmark import LEVELED_PROBLEMS, check_answer
from cognitive_swarm.evaluation.validate_baseline import bootstrap_ci, get_problems


PER_CATEGORY_INSTRUCTION = {
    "math":      "Solve it. Give ONLY the numeric answer, no explanation, no units work.",
    "reasoning": "Solve it. Give ONLY the final answer (a word, phrase, number, or 'No'), "
                 "no explanation.",
    "code":      "Say what this Python code outputs. Give ONLY the exact output value, "
                 "no explanation.",
    "trap":      "Carefully apply string/letter rules exactly. Give ONLY the final answer, "
                 "no explanation.",
}


def build_prompt(problem):
    cat = problem["category"]
    instr = PER_CATEGORY_INSTRUCTION.get(cat, "Give ONLY the final answer.")
    return (
        f"{instr}\n\n"
        f"Do NOT explain or show reasoning. Do NOT output chain-of-thought.\n"
        f"Problem: {problem['q']}\n\n"
        f"ANSWER: "
    )


def extract_answer(text: str) -> str:
    """Mirror the local _extract_plain: read the ANSWER: marker line."""
    if "ANSWER:" in text:
        idx = text.index("ANSWER:") + len("ANSWER:")
        return text[idx:].strip().split("\n")[0].strip()
    lines = [l for l in text.strip().split("\n") if l.strip()]
    return lines[-1].strip() if lines else text.strip()


class OpenRouterClient:
    def __init__(self, api_key=None, base="https://openrouter.ai/api/v1/chat/completions"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base = base
        self.available = bool(self.api_key)
        if not self.available:
            print("Warning: OPENROUTER_API_KEY not set.")

    def generate(self, prompt, model, max_tokens=40, temperature=0.0, retries=3):
        import httpx, time
        for attempt in range(retries):
            try:
                r = httpx.post(
                    self.base,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=60.0,
                )
            except Exception:
                time.sleep(3 + attempt * 3)
                continue
            if r.status_code == 200:
                try:
                    return r.json()["choices"][0]["message"]["content"]
                except Exception:
                    return None
            # rate-limited / transient provider error -> back off and retry
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 + attempt * 3)
                continue
            r.raise_for_status()
        raise RuntimeError(f"{model} failed after {retries} attempts (rate-limited or provider error)")


class BigModelSolver:
    def __init__(self, model="minimax/minimax-m3:free", client=None, temperature=0.0):
        self.model = model
        self.client = client or OpenRouterClient()
        self.temperature = temperature
        self.available = self.client.available

    def solve(self, q, problem=None, max_tokens=40):
        prompt = build_prompt(problem)
        text = self.client.generate(prompt, self.model,
                                    max_tokens=max_tokens, temperature=self.temperature)
        return extract_answer(text)


def run_big(subset, model, max_tokens, temperature, out, retries=2):
    client = OpenRouterClient()
    if not client.available:
        print("No OPENROUTER_API_KEY. Set it and retry.")
        return
    solver = BigModelSolver(model=model, client=client, temperature=temperature)
    problems = get_problems(subset)
    start = time.time()
    rows = []
    for p in problems:
        ok = False
        for attempt in range(retries + 1):
            try:
                ans = solver.solve(p["q"], problem=p, max_tokens=max_tokens)
                ok = check_answer(ans, p["a"])
                break
            except Exception as e:
                print(f"    (retry {attempt+1} {p['id']}: {e})", flush=True)
                time.sleep(2)
        rows.append({"id": p["id"], "level": p["level"], "category": p["category"],
                     "expected": p["a"], "solved_any": ok})
        print(f"  [{'OK' if ok else 'xx'}] {p['id']} [{p['level']}/{p['category']}] "
              f"({time.time()-start:.0f}s)", flush=True)
    stats = report_big(rows, subset)
    if out:
        json.dump({"model": model, "subset": subset, "stats": stats, "rows": rows},
                  open(out, "w"), indent=2)
        print(f"\nsaved -> {out}")


def report_big(rows, subset):
    n = len(rows)
    total = sum(r["solved_any"] for r in rows)
    accs = [1.0 if r["solved_any"] else 0.0 for r in rows]
    lo, hi = bootstrap_ci(accs)
    print("\n" + "=" * 64)
    print(f"BIG MODEL subset={subset} problems={n}")
    print(f"OVERALL 'solved in any run': {total}/{n} = {total/n:.1%}  "
          f"(bootstrap 95% CI [{lo:.1%}, {hi:.1%}])")
    for key_name, key_fn in [("level", lambda r: r["level"]),
                             ("category", lambda r: r["category"])]:
        groups = {}
        for r in rows:
            groups.setdefault(key_fn(r), []).append(r)
        parts = []
        for g, gs in sorted(groups.items()):
            gok = sum(x["solved_any"] for x in gs)
            parts.append(f"{g}={gok}/{len(gs)} ({gok/len(gs):.0%})")
        print(f"{key_name:9}: " + "  ".join(parts))
    print("=" * 64)
    return {"total": total, "n": n, "acc": total / n, "ci_low": lo, "ci_high": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minimax/minimax-m3:free")
    ap.add_argument("--subset", default="balanced")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_big(args.subset, args.model, args.max_tokens, args.temperature, args.out)


if __name__ == "__main__":
    main()
