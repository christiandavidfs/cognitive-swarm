# AGENTS.md — Cognitive Swarm

Guidance for working in this repo: the architecture, the hard-won empirical
findings, and the operational details that matter. Read this before changing
code or running experiments.

## What this project is

A local "cognitive swarm": small open models (Phi-3.5 Mini + Qwen 2.5 3B) on an
**8GB MacBook (M1)** that debate each other and are coordinated to beat a
single model — and, we've shown, to **beat their own "oracle"** (best-of-members)
via iterative cross-critique. Research question: *can small local models rival
a large cloud model through structured interaction?*

## The single most important empirical finding

**The swarm's value is NOT "averaging weak models".** It is (a) *recovering ~100%
of its combined (oracle) capability* with a good consensus, and (b) *EXCEEDING
the oracle* when models cross-critique each other's reasoning (emergence). The
remaining bottleneck is **SELECTION** (majority voting throws away novel-correct
minority answers), not generation.

## Architecture (the truth hierarchy — newest and most important)

Real learning doesn't rely on consensus; it uses the **cheapest, most-certain
source of truth** and a **verified memory** of experience. Tier order:

```
Tier 0  MEMORY    : verified (question -> {answer, confidence, tier, sources,
                   disagreement}) history. Instant, compounds, no compute.
Tier 1  CODE EXEC : code question -> run the interpreter (ground truth).
Tier 2  CALC      : simple arithmetic -> evaluate deterministically.
Tier 3  RETRIEVAL : factual/contested -> corroborate reliability-weighted
                   sources; surface CONFLICT honestly (history: different
                   sources, different versions -> no forced single answer).
Tier 4  DEBATE    : weak-model debate = HYPOTHESIS source (fallback only).
```

- **Consensus is a fallback, not the primary mechanism.** Use ground truth
  whenever it exists.
- **Never force a single answer for contested/ambiguous domains.** Return
  `(answer, confidence, sources, disagreement)` — provenance-aware resolution.
- Every verified answer is archived to memory so experience **accumulates**
  (repeats/rephrases hit memory in ms with zero inference).
- Modules: `cognitive_swarm/tools/executor.py` (sandboxed code exec),
  `cognitive_swarm/memory/verified_memory.py` (provenance-aware store),
  `cognitive_swarm/orchestration/truth_router.py` (tier routing),
  `cognitive_swarm/orchestration/corroboration.py` (reliability-weighted
  agreement + conflict handling).

## Key empirical results (so we don't re-discover them)

| Experiment | Result |
|---|---|
| Optimization of prompts (constraints + output format) | took same small models from ~0% to ~30% |
| Confidence-weighted consensus (v3) | **matches oracle** (61% vs 33% old on balanced 18-problem subset) |
| Cross-critique debate (v4) | **beats oracle**: hard 8-problem subset oracle 12% → system 38%, candidate pool 62% |
| Code executor (v5) | **100% on code** (was 8%); deterministic, 0 model loads |
| Memory compounding | pass 2 resolves 15/36 in 5.3ms; corroboration handles history conflicts honestly |
| **120-problem benchmark (credibility phase)** | **truth 44.2% [35.0-53.3] / phi 20.8% [14.2-28.3] / qwen 30.0% [21.7-38.3] / big (minimax-m3) 87.5% [81.7-92.5]**; truth = 100% code + 100% trap via deterministic tools (executor + string-op primitives), 0% reasoning, 0 model loads |
| **Open-response robustness (ORR)** | phi 2/6, qwen 1/6: both handle absurdism; neither has a stable "self" (song Jaccard=0); phi acknowledges a distressed user, qwen answers only the factual part |

## Open items / next (turn these into tasks when picked up)

- Run **debate** config on the full 120 for a true small-swarm vs frontier number.
- **String-op micro-primitives** (`count_char`, `count_vowels`, `count_letters`,
  `reverse`, `compare`) — DONE, land truth tier's trap 0% → 100% (12/12) at zero
  model loads, lifting the truth overall to 44.2% (was 34.2%).
- **Procedure / primitive memory**: cache reasoning *processes* (verified
  lemmas, winning-debate step-skeletons, discovered correction rules), not just
  answers; RAG corpus stays outside on disk.

## Operational facts (important)

- **Python**: `/Users/kaizen/repos/swarm/cognitive-swarm-env/bin/python3` (venv).
- **Models (MLX 4-bit)**: `mlx-community/phi-3.5-mini-instruct-4bit`,
  `mlx-community/Qwen2.5-3B-Instruct-4bit`. Loaded ONE at a time via
  `MultiModelBackend` (8GB RAM can't hold both simultaneously).
- **Sequential model loading is the dominant latency** (~10-15s per load-
  switch). Design experiments to minimize model loads: batch generations while
  a model is loaded; prefer ground-truth tools (executor/calculator/memory) over
  debate wherever possible. Full 36-problem benchmark with 2 debate samples
  exceeds the 40-min tool limit — use balanced subsets or `--samples 1`.
- **Small models answer FIRST then echo/explain the prompt.** Use a tight token
  budget (math/reasoning ~10, code ~24) to get short clean answers; extract
  preferring the START of output. Large per-category budgets produce verbose
  garbage that breaks regex extraction.
- **Big-model comparison: use free OpenRouter models, NOT Groq.** Groq's
  `qwen3.6-27b` emits a persistent *thinking-mode* block that (a) eats the token
  budget and (b) mangles naive extraction — it was the original cause of the
  8% under-count. Instead use `cognitive_swarm/evaluation/big_model_benchmark.py`
  against **`minimax/minimax-m3:free`** (OpenRouter token `sk-or-...` NOT in env
  by default; pass `OPENROUTER_API_KEY=` at runtime; never commit the token).
  Give it a tight answer-first prompt + `ANSWER:` marker extraction so it is
  scored fairly (verified: minimax-m3 87.5% on the 120 — its 2 balanced misses
  are the same ones gpt-4o misses, i.e. genuine reasoning limits).
- **Results are non-deterministic** on single runs; report "solved in any run".
  Some answers are latently known by a model but a fragile harness drops them —
  the debate's richer pipeline surfaces them.

## Memory-lookup correctness warning

Never let distinct technical questions blur together via token overlap — two
different code expressions share the same digits. VerifiedMemory is **code
aware**: for executable questions it requires an IDENTICAL normalized code
signature, and `extract_python` returns `None` for plain prose (via
`_looks_like_code`). If you change extraction, re-run
`truth_router_test.py --fresh` to confirm: code stays 12/12 and the history
conflict case still returns no forced answer.

## Conventions

- Deterministic tools (executor/calculator) and memory must run with **zero
  model loads** and be fast (ms).
- Structured resolutions carry provenance, never a bare atom.
- Docs live in `docs/` (RESULTS.md, CONCLUSIONS.md, ARCHITECTURE.md). Keep the
  honest-limits sections honest — the project's credibility depends on it.
- Run `python -m py_compile` on changed modules; there is no linter config.
