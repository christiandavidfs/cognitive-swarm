"""Validate the new confidence-weighted consensus on a balanced subset.

Runs the new OptimizedDebate (samples_per_model=2) on a representative subset
spanning all levels and categories, and compares its solve rate against:
  - the old consensus baseline (39% on full set)
  - the oracle (best-of both singles)

Writes incremental progress to a log so results are never lost.
"""
import sys, time, json
sys.path.insert(0, '.')
from pathlib import Path

from cognitive_swarm.orchestration.optimized_debate import OptimizedDebate
from cognitive_swarm.evaluation.leveled_benchmark import LEVELED_PROBLEMS, check_answer

# Balanced subset: 18 problems, 6 per level, all 3 categories.
subset_ids = [
    # easy
    'e-m-1','e-m-3','e-r-1','e-r-3','e-c-3','e-c-4',
    # medium
    'm-m-1','m-r-2','m-r-4','m-c-1','m-c-2','m-c-4',
    # hard
    'h-m-1','h-r-1','h-r-3','h-c-1','h-c-2','h-c-4',
]
probs = {p['id']: p for p in LEVELED_PROBLEMS}

# Load old results for oracle (best-of singles) baseline on this subset.
old = json.load(open('leveled_benchmark_results.json'))
def oracle_solves(pid):
    r = old['results'][pid]
    return any(x['correct'] for x in r.get('phi', [])) or any(x['correct'] for x in r.get('qwen', []))
def old_debate_solves(pid):
    r = old['results'][pid]
    return any(x['correct'] for x in r.get('debate', []))

debate = OptimizedDebate(samples_per_model=2)

log = open('new_consensus_validation.log', 'w')
results = {}
start = time.time()
for i, pid in enumerate(subset_ids):
    p = probs[pid]
    r = debate.run_debate(p['q'], max_tokens=32)
    ok = check_answer(r.final_answer, p['a'])
    results[pid] = {'ok': ok, 'final': r.final_answer, 'conf': r.metadata['confidence'],
                    'num_votes': r.metadata['num_votes']}
    mark = 'OK' if ok else 'xx'
    line = f"{mark} {pid} [{p['level']}/{p['category']}] exp={p['a']!r:>8} -> {r.final_answer[:22]!r:<24} conf={r.metadata['confidence']}"
    print(line, flush=True)
    log.write(line + '\n'); log.flush()
    elapsed = time.time() - start
    print(f"  [{i+1}/{len(subset_ids)}] elapsed={elapsed:.0f}s", flush=True)

# Aggregate
new_solve = sum(1 for pid in subset_ids if results[pid]['ok'])
oracle_solve = sum(1 for pid in subset_ids if oracle_solves(pid))
old_solve = sum(1 for pid in subset_ids if old_debate_solves(pid))

print()
print('=' * 70)
print(f'Subset size: {len(subset_ids)}')
print(f'NEW consensus (samples=2)\t: {new_solve}/{len(subset_ids)} = {new_solve/len(subset_ids):.0%}')
print(f'OLD debate heuristic (recorded)\t: {old_solve}/{len(subset_ids)} = {old_solve/len(subset_ids):.0%}')
print(f'ORACLE (best of phi|qwen)\t: {oracle_solve}/{len(subset_ids)} = {oracle_solve/len(subset_ids):.0%}')
print()
print('Improvement over old:', new_solve - old_solve, 'problems')
print('Gap to oracle:', oracle_solve - new_solve, 'problems')
print('=' * 70)

json.dump(results, open('new_consensus_subset.json', 'w'), indent=2)
