"""Validation harness: run a solver config on the benchmark with repeats and
report mean accuracy per level/category plus a bootstrap 95% confidence
interval, so results are statistically measurable rather than one lucky run.

Configs:
  phi, qwen, debate (OptimizedDebate), emergent (EmergentDebate + reasoned
  selector), truth (TruthRouter: executor+memory, 0 model loads).

Usage:
  python -m cognitive_swarm.evaluation.validate_baseline --config debate \
      --subset all --runs 1
  python -m cognitive_swarm.evaluation.validate_baseline --config truth \
      --subset all
  python -m cognitive_swarm.evaluation.validate_baseline --config emergent \
      --subset hard --runs 1     # slowest: ~6 model loads/problem
"""
import sys, json, random, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognitive_swarm.evaluation.leveled_benchmark import (
    LEVELED_PROBLEMS, check_answer,
)


def get_problems(subset):
    if subset == "all":
        return LEVELED_PROBLEMS
    if subset == "balanced":
        # 12 per level, representative across categories + a few trap.
        return [p for p in LEVELED_PROBLEMS if p["level"] != "trap"][:0] or \
               [p for p in LEVELED_PROBLEMS
                if p["id"] in {
                    "e-m-1","e-m-3","e-r-1","e-r-3","e-c-3","e-c-4",
                    "m-m-1","m-r-2","m-r-4","m-c-1","m-c-2","m-c-4",
                    "h-m-1","h-r-1","h-r-3","h-c-1","h-c-2","h-c-4",
                }]
    if subset == "hard":
        return [p for p in LEVELED_PROBLEMS if p["level"] == "hard"]
    if subset == "trap":
        return [p for p in LEVELED_PROBLEMS if p["category"] == "trap"]
    if subset == "easy":
        return [p for p in LEVELED_PROBLEMS if p["level"] == "easy"]
    if subset == "medium":
        return [p for p in LEVELED_PROBLEMS if p["level"] == "medium"]
    if subset == "non_code":
        return [p for p in LEVELED_PROBLEMS if p["category"] != "code"]
    if subset.startswith("ids:"):
        ids = set(subset[4:].split(","))
        return [p for p in LEVELED_PROBLEMS if p["id"] in ids]
    return LEVELED_PROBLEMS


def make_solver(config):
    if config == "phi":
        from cognitive_swarm.evaluation.leveled_benchmark import PhiSolver
        return PhiSolver()
    if config == "qwen":
        from cognitive_swarm.evaluation.leveled_benchmark import QwenSolver
        return QwenSolver()
    if config == "debate":
        from cognitive_swarm.evaluation.leveled_benchmark import OptimizedDebateSolver
        return OptimizedDebateSolver(samples_per_model=1)
    if config == "emergent":
        from cognitive_swarm.orchestration.emergent_debate import EmergentDebate
        return EmergentDebate(answer_samples=1, critique_samples=1)
    if config == "truth":
        from cognitive_swarm.orchestration.truth_router import TruthRouter
        router = TruthRouter()
        from cognitive_swarm.orchestration.truth_router import demo_fact_table
        demo_fact_table(router)
        return router
    raise ValueError(f"unknown config: {config}")


def solve(solver, config, problem):
    q = problem["q"]
    if config == "emergent":
        return solver.run(q).final_answer
    if config == "truth":
        res = solver.resolve(q)
        return (res.answer or "") if hasattr(res, "answer") else (res or "")
    return solver.solve(q)


def bootstrap_ci(accs, n_boot=2000, seed=0, alpha=0.95):
    """Bootstrap 95% CI over a list of per-problem 0/1 accuracies."""
    if not accs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(accs)
    means = []
    for _ in range(n_boot):
        s = sum(rng.choice(accs) for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int((1 - alpha) / 2 * n_boot)]
    hi = means[int((1 + alpha) / 2 * n_boot) - 1]
    return (lo, hi)


def run_validation(config, subset, runs, seed):
    problems = get_problems(subset)
    solver = make_solver(config)
    start = time.time()
    rows = []
    for p in problems:
        per_run = []
        for r in range(runs):
            ans = solve(solver, config, p)
            per_run.append(check_answer(ans, p["a"]))
        solved_any = any(per_run)
        rows.append({
            "id": p["id"], "level": p["level"], "category": p["category"],
            "expected": p["a"], "runs": per_run, "solved_any": solved_any,
        })
        mark = "OK" if solved_any else "xx"
        print(f"  [{mark}] {p['id']} [{p['level']}/{p['category']}] {per_run} "
              f"({time.time()-start:.0f}s)", flush=True)
    return rows


def report(rows, config, subset, runs):
    n = len(rows)
    total = sum(r["solved_any"] for r in rows)
    accs = [1.0 if r["solved_any"] else 0.0 for r in rows]
    lo, hi = bootstrap_ci(accs)
    print("\n" + "=" * 64)
    print(f"config={config} subset={subset} problems={n} runs/problem={runs}")
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
    return {"total": total, "n": n, "acc": total / n,
            "ci_low": lo, "ci_high": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["phi", "qwen", "debate", "emergent", "truth"])
    ap.add_argument("--subset", default="all")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = run_validation(args.config, args.subset, args.runs, args.seed)
    stats = report(rows, args.config, args.subset, args.runs)
    if args.out:
        json.dump({"config": args.config, "subset": args.subset,
                   "stats": stats, "rows": rows},
                  open(args.out, "w"), indent=2, default=str)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
