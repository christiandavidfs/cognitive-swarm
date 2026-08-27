"""Experiment: does an iterative cross-critique debate EXCEED the oracle?

Plots a decisive test of the hive-mind premise: a swarm is only worth building
if it can solve problems that NEITHER single model can solve — i.e. beat the
oracle (best of phi|qwen).

For a focused subset we run:
  * phi alone, qwen alone  -> define the oracle
  * the emergent debate    -> swarm system answer + candidate pool
and report:

  - oracle accuracy
  - swarm system accuracy (the consensus final answer)
  - swarm candidate accuracy (correct answer present anywhere in the pool)
  - EMERGENCE: problems where oracle FAILS but the swarm's pool/system has the
    answer. This is the only place the swarm can justify its existence.

Usage:  python -m cognitive_swarm.evaluation.emergence_test [n_problems]
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognitive_swarm.evaluation.leveled_benchmark import LEVELED_PROBLEMS, check_answer, PhiSolver, QwenSolver
from cognitive_swarm.orchestration.emergent_debate import EmergentDebate


def analyze_subset(problems):
    phi_solver = PhiSolver()
    qwen_solver = QwenSolver()
    debate = EmergentDebate(answer_samples=1, critique_samples=1)

    rows = []
    for p in problems:
        pid, q, exp = p["id"], p["q"], p["a"]

        phi_ans = phi_solver.solve(q)
        qwen_ans = qwen_solver.solve(q)

        res = debate.run(q)

        phi_ok = check_answer(phi_ans, exp)
        qwen_ok = check_answer(qwen_ans, exp)
        oracle_ok = phi_ok or qwen_ok
        swarm_ok = check_answer(res.final_answer, exp)

        # candidate pool: any extracted candidate correct?
        pool = [a for lst in res.candidates.values() for a in lst]
        cand_ok = any(check_answer(a, exp) for a in pool)

        rows.append({
            "id": pid, "exp": exp,
            "phi": phi_ans, "phi_ok": phi_ok,
            "qwen": qwen_ans, "qwen_ok": qwen_ok,
            "oracle_ok": oracle_ok,
            "swarm": res.final_answer, "swarm_ok": swarm_ok,
            "candidates": res.candidates,
            "candidate_ok": cand_ok,
            "conf": res.confidence,
        })
        status = "O" if oracle_ok else ("E" if (cand_ok or swarm_ok) else ".")
        print(f"[{status}] {pid} exp={exp!r:>16} | phi={phi_ans!r:<14}{'OK' if phi_ok else '--':<3} "
              f"qwen={qwen_ans!r:<14}{'OK' if qwen_ok else '--':<3} | "
              f"swarm={res.final_answer!r:<14}{'OK' if swarm_ok else '--':<3} "
              f"cand={cand_ok} conf={res.confidence}")
        sys.stdout.flush()

    return rows


def report(rows):
    n = len(rows)
    oracle = sum(r["oracle_ok"] for r in rows)
    swarm_sys = sum(r["swarm_ok"] for r in rows)
    swarm_cand = sum(r["candidate_ok"] for r in rows)

    # Emergence: oracle fails but swarm gets it.
    emerg_sys = [r for r in rows if not r["oracle_ok"] and r["swarm_ok"]]
    emerg_cand_only = [r for r in rows if not r["oracle_ok"] and r["candidate_ok"] and not r["swarm_ok"]]

    print("\n" + "=" * 62)
    print(f"Subset size           : {n}")
    print(f"Oracle (phi|qwen)     : {oracle}/{n} = {oracle/n:.0%}")
    print(f"Swarm system (consensus): {swarm_sys}/{n} = {swarm_sys/n:.0%}")
    print(f"Swarm candidate pool  : {swarm_cand}/{n} = {swarm_cand/n:.0%}")
    print("-" * 62)
    print(f"EMERGENCE (oracle fails, swarm solves):")
    print(f"  system-level  : {[r['id'] for r in emerg_sys]}")
    print(f"  candidate-only: {[r['id'] for r in emerg_cand_only]}")
    print("=" * 62)

    return {
        "n": n,
        "oracle": oracle,
        "swarm_system": swarm_sys,
        "swarm_candidate": swarm_cand,
        "emergence_system": [r["id"] for r in emerg_sys],
        "emergence_candidate_only": [r["id"] for r in emerg_cand_only],
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    probs = {p["id"]: p for p in LEVELED_PROBLEMS}

    # Focus on problems where emergent value is most plausible (hard + some
    # medium), plus a few easy controls. Fixed list = reproducible.
    subset_ids = [
        "h-r-1", "h-r-3", "h-r-4", "h-m-1", "h-m-3", "h-m-4",
        "m-r-4", "m-r-3", "m-m-2", "m-m-4",
        "e-m-1", "e-r-4",
    ][:n]

    problems = [probs[i] for i in subset_ids]
    rows = analyze_subset(problems)
    stats = report(rows)
    out = {"problems": rows, "stats": stats}
    Path("emergence_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved emergence_results.json")
