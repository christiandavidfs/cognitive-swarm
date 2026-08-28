"""Demonstrate the real-life truth hierarchy across all tiers.

   1. Deterministic tiers: code executor + calculator (0 model loads).
   2. Memory compounding: repeat/rephrased questions hit memory instantly.
   3. Corroboration (Tier 3): reliability-weighted agreement.
   4. Conflict handling: reliable sources disagree -> honest disagreement, not
      a forced single answer.

Usage:  python -m cognitive_swarm.evaluation.truth_router_test
"""

import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognitive_swarm.evaluation.leveled_benchmark import LEVELED_PROBLEMS, check_answer
from cognitive_swarm.memory.verified_memory import VerifiedMemory
from cognitive_swarm.orchestration.truth_router import TruthRouter, demo_fact_table


def run_deterministic_pass(router, problems, label):
    t0 = time.time(); resolved = 0; used = {}
    by_cat = {"math": [0, 0], "reasoning": [0, 0], "code": [0, 0], "trap": [0, 0]}
    for p in problems:
        r = router.resolve(p["q"])
        if r.answer is not None:
            resolved += 1
            used[r.status] = used.get(r.status, 0) + 1
        ok = (r.answer is not None) and check_answer(r.answer, p["a"])
        by_cat[p["category"]][0] += ok
        by_cat[p["category"]][1] += 1
    dt = time.time() - t0
    print(f"\n[{label}] resolved={resolved}/{len(problems)} "
          f"time={dt:.2f}s (0 model loads)")
    for cat, (ok, tot) in by_cat.items():
        print(f"   {cat:10}: {ok}/{tot} = {ok/tot:.0%}")
    print(f"   sources: {used}")


def main():
    fresh = "--fresh" in sys.argv
    mem = VerifiedMemory(path=tempfile.mktemp(suffix=".json") if fresh else None)
    router = TruthRouter(memory=mem)
    demo_fact_table(router)

    print("=== PASS 1: cold memory, deterministic tiers only (no models) ===")
    run_deterministic_pass(router, LEVELED_PROBLEMS, "pass 1 (cold)")

    print("\n=== PASS 2: same questions -> memory hits (no recompute) ===")
    t0 = time.time(); hits = 0
    for p in LEVELED_PROBLEMS:
        r = router.resolve(p["q"])
        if r.status == "memory": hits += 1
    print(f"   memory hits: {hits}/{len(LEVELED_PROBLEMS)}  "
          f"time={ (time.time()-t0)*1000:.1f}ms total")

    print("\n=== SIMILARITY regression: distinct code questions must NOT match ===")
    mem2 = VerifiedMemory(path=tempfile.mktemp(suffix=".json"))
    r2 = TruthRouter(memory=mem2)
    r2.resolve("What does this Python code output: print(10 // 3)?")   # -> 3
    wrong = r2.resolve("What does this Python code output: print(2 ** 3 ** 2)?")
    print(f"   stored 10//3=3; lookup print(2**3**2) -> {wrong.answer!r} "
          f"(must be None, not 3). status={wrong.status}")

    print("\n=== TIER 3 corroboration (reliability-weighted) ===")
    r3 = router.resolve("When did the Fall of the Western Roman Empire happen?")
    print(f"   Q: 'Fall of the Western Roman Empire'")
    print(f"   => {r3.answer!r}  confidence={r3.confidence}  status={r3.status}")
    for s in r3.sources[:2]:
        print(f"      source: {s}")

    print("\n=== TIER 3 conflict (reliable sources disagree -> honest) ===")
    r4 = router.resolve("What caused the fall of the Roman Empire?")
    print(f"   Q: 'what caused the fall of the roman empire'")
    print(f"   => answer={r4.answer!r}  status={r4.status}  (no forced single answer)")
    for d in r4.disagreement:
        print(f"      disagreement: {d['answer']!r}  from {d['sources']}  "
              f"(reliability {d['best_reliability']})")

    print(f"\nMemory size after demo: {mem.size()} entries")
    print("memory file:", mem.path)


if __name__ == "__main__":
    main()
