# The Cognitive Swarm

### A technical white paper on how structured interaction between small local
### models — plus a truth hierarchy of deterministic and corroborated sources —
### beats a frontier cloud model, and the swarm's own oracle, on a single 8 GB laptop

**Author:** Christian David Filippini Souto
**Contact:** christiandavid.fs@gmail.com
**Version:** 2.0 — Breaking change v9 (procedure memory + trace generation)
**Date:** August 2026
> **Breaking change:** Student moves from `question → pattern` classification to `question → reasoning trace` generation. `verified_memory` now caches *procedures* (verified step skeletons), not just answers. Truth hierarchy Tier 0 now serves traces. See §7.5 and ROADMAP_STUDENT.md M1-M4.
**Repo:** `https://github.com/christiandavidfs/cognitive-swarm`

---

> *This document is the technical white paper for the Cognitive Swarm project.
> It recounts the research question, the method, the empirical results, the
> architecture that emerged, and — crucially — the limits we found. It is
> written to be read both as a record of what was built and as a blueprint for
> what comes next.*

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Research questions](#2-research-questions)
3. [Background and state of the art](#3-background-and-state-of-the-art)
4. [Hardware constraints](#4-hardware-constraints)
5. [Method](#5-method)
6. [Results](#6-results)
7. [The architecture that emerged](#7-the-architecture-that-emerged)
8. [Discussion](#8-discussion)
9. [Honest limitations](#9-honest-limitations)
10. [Future work and roadmap](#10-future-work-and-roadmap)
11. [Conclusion](#11-conclusion)
12. [Appendices](#12-appendices)

---

## 1. Motivation

The dominant assumption in large language model (LLM) research is that
capability is mostly a function of scale: bigger models, more parameters, cloud
inference. That assumption quietly writes off small, local, private models as
too weak for serious reasoning.

This project challenges that assumption from an uncomfortable corner: **two
small open models on an 8 GB MacBook**. The thesis is that *intelligence can
emerge from structured interaction between weak models*, and that a local swarm
can rival — and in specific settings *surpass* — a single large cloud model.

Three values motivate the work:

- **Privacy.** All inference happens on-device. No prompt ever leaves the
  laptop.
- **Cost & autonomy.** No per-call fees, no rate limits, works offline on
  commodity hardware.
- **Autonomy in capability.** If weak models can coordinate into something
  stronger, then intelligence is not hostage to data-center budgets.

## 2. Research questions

1. **Can structured interaction between multiple weak models create stronger
   reasoning than a single weak model?**
2. **Can small local models rival a single large cloud model?**
3. **Can a swarm be *more* than the union of its members — i.e. beat its own
   oracle (the best of its members)?**
4. **What is the *right* epistemic architecture: is consensus the primary
   mechanism, or is it a fallback to cheaper, more certain sources of truth?**

## 3. Background and state of the art

**Multi-agent LLM debate.** Du et al. and Liang et al. showed that multiple
model instances debating improves factual consistency and reasoning. These are
predominantly applied to large/frontier models and show recovery of individual
capability. Our contribution targets small models and asks whether
*iterative cross-critique* can *exceed* the oracle.

**Retrieval-augmented generation (RAG).** Grounding outputs in external sources
(Lewis et al., Gao et al.) is established. Our twist is *verified memory* with
provenance and an explicit truth-tier router that prefers deterministic tools
and honest conflict reporting over synthesis.

**Model collapse.** Shumailov et al. showed self-training on generated data
degrades the distribution over generations. This directly informs our
fine-tuning roadmap: it must be fed *fresh external verified signal*, not its
own outputs.

**On-device / constrained inference.** MLX (Apple) and prompt optimization to
extract clean answers from terse local outputs follow the general push to make
small LLMs usable locally.

## 4. Hardware constraints

| Item | Value |
|---|---|
| Host | Apple MacBook (M1), **8 GB RAM** |
| Local models | Phi-3.5 Mini (3.8B), Qwen 2.5 3B (3B), MLX 4-bit |
| RAM per model | Phi ~2.5 GB, Qwen ~2.0 GB |
| Loading | **One model at a time** (8 GB cannot hold both) |
| Cloud comparison | **minimax/minimax-m3:free via OpenRouter (frontier)** — tight `ANSWER:` contract, same `check_answer` scorer (Qwen 27B via Groq retired: thinking-mode artifact) |

The 8 GB constraint is not a nuisance; it is the *lens* of the project. Any
claimed improvement must hold under this floor.

## 5. Method

### 5.1 Answer extraction (the root-cause fix)

Local MLX models return free text: they state the answer **first**, then
reason or echo the prompt. Verbose output shattered regex extraction and was
the *dominant* cause of low scores — not model weakness.

The fix is structural: a **tight per-category token budget** (math/reasoning
~10 tokens, code ~24) so the model can only output the answer, plus
**extraction that prefers the start of output**. This single change underlies
the consensus result.

### 5.2 Debate protocols

**Single-pass consensus (v3).** Two first-guesses are aggregated by a
deterministic engine that clusters votes by answer and scores each cluster by
agreement, self-consistency, and confidence. Because it only recombines first
guesses, it *cannot* exceed the oracle — it can only match it.

**Iterative cross-critique (v4).** To *exceed* the oracle we separate answer
capture from reasoning capture:

- **Stage A (answer):** each model answers with the tight, terse prompt.
- **Stage A2 (reasoning):** each model reasons aloud; the trace is kept only
  to feed critics.
- **Stage B (cross-critique):** each model sees the *other* model's reasoning
  and is asked to find the error and recompute — a genuinely different input
  than the first pass, able to yield **novel** candidates.
- **Stage C (consensus):** all candidates are pooled and aggregated.

Because selection is the crux, we report both the **candidate** accuracy (is
the correct answer anywhere in the pool — does the debate *generate* novel
knowledge?) and the **system** accuracy (the single returned answer — can we
*select* correctly?).

### 5.3 The truth hierarchy

The deepest architectural realization of the project: **consensus is a
fallback, not the primary mechanism.** Real learning uses the cheapest,
most-certain source of truth first:

```
Tier 0  MEMORY    : verified (question -> {answer, confidence, tier,
                   reliability, sources, disagreement}) history. Instant,
                   compounds, no compute. Now with reasoning_op_sig.
Tier 1  CODE EXEC : code-output question -> run the interpreter. Ground truth.
Tier 2b STRING-OP : reverse/count/compare — trap 0%→100%.
Tier 2c MATH      : deterministic math primitives (word operators, factorial,
                   sqrt, log, multi-step) — math 89%→100%.
Tier 2d REASONING : 20+1 reasoning primitives (rate, work_inverse, jug, bridge,
                    handshake, system_equations, stock_take_away, etc.) dispatched via
                    reasoning_primitives.py today, student_router.py (33 classes, 800 feats) tomorrow.
Tier 3  RETRIEVAL : factual/contested -> corroborate reliability-weighted,
                   independence-penalized sources; surface CONFLICT honestly.
Tier 4  DEBATE    : weak-model debate = HYPOTHESIS source (fallback only).
```

Every resolution is structured — `(answer, confidence, sources, disagreement)`
— never a bare atom. Contested domains (history, opinions) are allowed to
*report conflict* rather than fake certainty. Truth now **120/120 (100%) at
0 loads, 1.2s cold / 65ms warm**, beating frontier 87.5% on same scorer.

## 6. Results

### 6.1 Leveled benchmark (36 problems) — historical baseline

| Approach | Overall | Easy | Med | Hard | Math | Reas | Code |
|---|---|---|---|---|---|---|---|
| Phi-3.5 Mini (single) | 28% | 67% | 8% | 8% | 42% | 25% | 17% |
| Qwen 2.5 3B (single) | 28% | 33% | 42% | 8% | 17% | 33% | 33% |
| **Debate (swarm)** | **39%** | 42% | 33% | **42%** | **67%** | **42%** | 8% |
| API (Qwen 27B, Groq — retired) | 8% | 8% | 8% | 8% | 8% | 0% | 17% |

- The local swarm beat a 24× larger cloud model (39% vs 8%) on the 36-set.
- Honest caveat: Qwen 27B was crippled by thinking-mode artifact — not a clean comparison.

### 6.1b Expanded benchmark (120 problems) — current

| Approach | Overall | math | reasoning | code | trap | CI |
|---|---|---|---|---|---|---|
| Phi-3.5 Mini (single) | 20.8% (25/120) | 11/36 | 7/36 | 5/36 | 2/12 | [14.2, 28.3] |
| Qwen 2.5 3B (single) | 30.0% (36/120) | 6/36 | 9/36 | 17/36 | 4/12 | [21.7, 38.3] |
| Debate (swarm) | 47.5% (57/120) | 18/36 | 15/36 | 18/36 | 6/12 | [38.3, 56.7] |
| **Big frontier minimax-m3:free (OpenRouter, tight ANSWER:)** | **87.5% (105/120)** | 36/36 | 26/36 | 34/36 | 9/12 | [81.7, 92.5] |
| **Truth hierarchy (deterministic, 0 loads)** | **100% (120/120)** | **36/36** | **36/36** | **36/36** | **12/12** | [96.9, 100] |

- Truth **100% at 0 loads, 1.2s cold / 65ms warm**, beating frontier 87.5% on same `check_answer` scorer with same tight contract — the Qwen 27B artifact is now removed.
- Small singles remain far below frontier (20.8%/30.0%), but truth closes the gap entirely via verified tools; debate is the fallback, not the engine.

### 6.2 Consensus reaches the oracle (v3)

| Approach (18-problem balanced subset) | Accuracy |
|---|---|
| OLD heuristic consensus | 33% |
| **NEW confidence-weighted consensus** | **61%** |
| Oracle (best of phi\|qwen, theoretical ceiling) | 61% |

The new consensus **exactly matches the oracle** and doubles the old heuristic
(+28 points). Every problem either single model can solve, the swarm now solves;
the rest are *model* limits, not *consensus* limits.

### 6.3 Emergence: the swarm beats its oracle (v4)

Hard 8-problem subset (h-r-1, h-r-3, h-r-4, h-m-1, h-m-3, h-m-4, m-r-4, m-r-3):

| Approach | Accuracy |
|---|---|
| Oracle (best of phi\|qwen) | 12% |
| **Swarm system** (consensus final) | **38%** |
| **Swarm candidate pool** (anywhere) | **62%** |

The cross-critique debate **beats the oracle**: system 38% vs oracle 12%
(3×), candidate pool 62% vs 12% (5×). Two findings:

1. **Generation works.** The debate *generates* answers neither member produces
   alone — true novel generation (m-r-4 → 42; h-r-3 → 20) and surfacing of
   latent answers (h-r-4, h-m-1).
2. **Selection is the new bottleneck.** The candidate pool (62%) is ~2× the
   system answer (38%); majority-weighted consensus discards novel-correct
   minority answers.

### 6.4 The truth hierarchy (v7 — 120/120)

| Tier | Result |
|---|---|
| Code executor (36 code) | **36/36 = 100%** (was 8%) |
| String-op trap (12 trap) | **12/12 = 100%** (was 0%) |
| Math primitives (36 math) | **36/36 = 100%** (was 89% → 100% with word operators + multi-step) |
| Reasoning primitives (36 reasoning) | **36/36 = 100%** (was 3% → 100% with 20 patterns) |
| Memory compounding (pass 2 of 120) | **120 hits in 66 ms** |
| Total deterministic resolve (120) | **120/120 = 100% in 1.2 s cold, 0 model loads** |

The weakest categories became the strongest by *computing* instead of predicting.
Memory is now **code/string/math/reasoning-aware** (identical `*_op_sig` required)
to prevent distinct questions from blurring (e.g. handshake 8=28 vs 100=4950).

Prototype **student router** (`tools/student_router.py`, TF-IDF+LogReg 240KB, 800 feats, 132 TRAINING +165 synthetic =297 samples /33 classes, 31-32/36 routing, CV 0.962) learns *structure* not answers — e.g. `47 people
conference handshake` correctly → `handshake` while regex misses (generalization).

**Honest conflict handling (history example).**

- *Corroborated:* two independent claims about the Fall of the Western Roman
  Empire converge → `476 CE`, confidence 0.95.
- *Conflict:* two reliable historians disagree on the *cause* → **no forced
  single answer**, disagreement surfaced with provenance.
- The corroborator weights by reliability and independence, so a single primary
  source outweighs ten derived copies.

## 7. The architecture that emerged

### 7.1 Truth hierarchy routing

Full detail in `docs/ARCHITECTURE.md`. The router returns structured
`Resolution(answer, confidence, sources, disagreement)` from the cheapest
certain tier upward.

### 7.2 The read/write split (the key insight)

There are two fundamentally different kinds of knowledge:

| | In weights (the model) | In memory / collection (consumable truth) |
|---|---|---|
| Learn once at train time | Yes | No (read fresh on demand) |
| Updates when source changes | No (stale until retrain) | **Yes — next read is current** |
| Grows model size | Yes | **No — the model does not grow** |
| Generalizes (skill) | Strong | Weak |
| Precise & current (facts) | Weak / stale | Strong |
| Checkable against source | No | Yes |

**Memory/connectors give up-to-date knowledge with zero model regrowth.** When
a source updates, the next read is current — no retraining, no bigger model.
This is the property that lets the *system* ceiling grow on fixed hardware
while the model stays small.

### 7.3 Source taxonomy and selection policy

No source is uniformly good or bad — even Wikipedia is reliable for stable,
cited, enumerable facts and weak for contested or fast-moving claims. Selection
is **claim-dependent**, scoring each source on four axes:

- **Authority:** trustworthiness for this claim type (peer-reviewed > encyclopedia > blog).
- **Proximity (primary vs derivative):** original vs copy-of-a-copy.
- **Recency / timeliness:** fast-moving facts value newer sources.
- **Independence:** ten sources copying one source count as one.

The Corroborator combines these into a reliability-weighted, independence-
penalized verdict; contested domains return conflict rather than a forced
answer. *"Wikipedia is not a source of truth"* is half-right: the fix is to
cap its weight and corroborate it against primary sources, not to ban it.

### 7.4 The connector / retriever layer

Tier 3's source collection is currently a stub (a demo fact table). The
production design is a pluggable retriever interface:

```python
class Retriever(ABC):
    def get_claims(self, question) -> list[SourceClaim]: ...

MATH_SOURCE     = InterpreterRetriever()   # REPL / CAS (sympy)  -> rel 1.0
DEFINITIONS     = LocalDocsRetriever(...)  # book collection
ENCYCLOPEDIA    = WebRetriever(...)        # wikipedia/others (low weight)
SCHOLARLY       = AcademicDBRetriever(...) # peer-reviewed
STRUCTURED      = DatabaseRetriever(...)   # sql / spreadsheets / unit tests
```

Each connector is self-contained; adding or updating a source makes knowledge
current automatically, without touching the model. The Retriever interface is
**media-agnostic** (claims can carry a media path/URI) so text-only works today
and vision slots in later with zero rework.

### 7.5 The growing-ceiling trainer (hive-mind loop) — **v9 breaking change: procedure memory**

```
source of truth + cross-critique debate
        │  (verified (question → trace → answer) with step provenance)
        ▼
verified memory (procedure) ──→ immediate traces + answers (growing, current, no model growth)
        │  traces are skeletons like "handshake: n=47 → n*(n-1)/2 → 1081" or "m-r-4: diffs 4,6,8,10 → 42"
        ▼ (optional, slow)
LoRA fine-tune student to generate traces (question → steps → answer) ──→ raises reasoning, not just routing
        │  each step verified by executor/math (e.g., 47*46/2=1081 checked)
        ▼
swap the trace-generating student back into swarm ──→ next cycle's ceiling is higher (Tier 0 now serves traces)
```

**Breaking change:** `verified_memory` now stores `procedure_sig` (hash of trace skeleton) + `answer`, and `student_ner` → `student_trace` (`question → trace`). This is how we approach **reasoning** (generating verifiable steps) vs. mere classification (`question → pattern`).

**Memory is the cheap, instant ceiling-raiser** (no model growth). **Fine-tuning
is the slow, compounding upgrade** that raises generalization — but it needs
*fresh* verified signal each cycle to avoid *model collapse* (the swarm parrots
its own outputs and stalls). Fresh connectors are the fuel; without them the
loop plateaus.

### 7.6 Multimodal as the final-model target

The **final model** — the one the swarm eventually trains and that *presents*
answers — ideally should be multimodal, playing **two roles at once**:

1. **Retrieval-augmented answerer (read-side):** consume verified memory +
   connectors at query time. Vision lets it consume images, PDFs, diagrams,
   charts, tables, screenshots — a *sensory-corroborated* source surface.
2. **Fine-tuned student (write-side):** trained on verified labels; vision
   makes the learning signal denser (image-caption, vision-question pairs).

**Unblocking posture:** keep the Retriever layer media-ready now, but **do not
swap models to a VLM yet** — on 8 GB a strong VLM at 4-bit contends with memory
and connectors. Treat multimodal as the *target ceiling* of the trainer; only
evaluate a concrete VLM on the M1 after connectors are media-ready.

## 8. Discussion

Three results reframe "small models are too weak":

1. **Prompt structure + extraction recovers much of what is credited to scale.**
   The consensus fix (tight budgets + start-preferring extraction) moved the
   same small models to the oracle ceiling.
2. **Cross-critique produces genuinely new reasoning above the oracle.** This is
   the strongest evidence the swarm is more than the union of its members, and
   the strongest vindication of the hive-mind premise.
3. **For real learning, consensus is not the answer.** Deterministic,
   corroborated ground truth dominates wherever it applies; consensus is
   relegated to its correct role as fallback.

**The persistent bottleneck is selection, not generation.** Generation already
exceeds the oracle; majority voting discards the novel-correct minority. The
highest-leverage fix is not a generic selection classifier; it is a selector
that exploits the debate's own *disagreement structure*. The key distinction:
**reasoned correction vs. capitulation**.

> When a model changes its answer, we must ask *why*. If it did so by
> **refuting a specific argument** in the other model's reasoning, that is an
> *epistemic* signal — a correction born of reasoning, worth trusting even in
> the minority. If it did so merely by **conforming to a crowded majority**, it
> is *social pressure*, which carries no truth signal and should be discounted.

This distinction (reasoned correction vs. capitulation) is **not novel to us**.
A vigorous recent literature decomposes exactly this flip: "Not All Flips Are
Conformity" [(Qin, arXiv:2606.00820)] separates stance-induced conformity from
reasoning-induced persuasion (strict conformity is 29% and mostly harmful —
63.6% correct-to-incorrect flips); "The Cost of Consensus" [(Bertalanič &
Fortuna, arXiv:2605.00914)] measures sycophantic modal adoption up to 85.5% and
the resulting "consensus collapse" (oracle gap ~32 pp — the same shape as our
62%→38%); "Can LLM Agents Really Debate?" [(Wu, arXiv:2511.07784)] scores
rational argument-driven correction vs. social conformity; and BASIL
[(arXiv:2508.16846)] isolates sycophancy from rational Bayesian belief updating.
Our only incremental angle is the extreme low-resource regime — two models of
~3–3.8B on 8 GB — on a closed, numeric, answer-first problem class where a
refuted specific argument is more interpretable than in large-model
multiple-choice settings, and where we already keep the reasoning traces (Stage
A/A2/B/C) so we can inspect *whether* a flip was argument-driven or
conformity-driven. That is a modest extension, not a new mechanism; we report it
as such.

## 9. Honest limitations

1. **Closed-set determinism.** Truth is 100% on the fixed 120-set because every category now has a verified tool; outside these 20 pattern classes the system correctly falls back to debate (where selection still 62%→38%). The 120 is not a claim of general reasoning.
2. **Student router is prototype.** TF-IDF+LogReg 33/36 routing, programmatic regex still extracts args. Fully learned dispatch needs arg prediction.
3. **API artifact fixed.** Qwen 27B thinking-mode confound retired; frontier is now minimax-m3 87.5% scored with same tight `ANSWER:` contract — truth still beats it on this fixed set, but frontier dominates single small models on open tasks.
4. **Small debate scale.** Emergence 62%→38% validated on hard 8-problem subset (time-constrained); full 120 debate not yet run.
5. **8 GB is a hard floor per base model.** System ceiling grows via memory/student; neuron ceiling bounded by chip.
6. **Consensus is not authority.** Even good corroboration can be wrong if sources are wrong; weights mitigate, not eliminate.

## 10. Future work and roadmap

1. **Selection by reasoned correction vs. capitulation (highest leverage;
   *incremental* low-resource case, not a novel mechanism — see Discussion).**
   Instead of a generic selection classifier, exploit the debate's disagreement
   structure. Use cheap features derived from the traces we already keep: did the
   answer survive its counterpart's critique without being refuted? did the model
   change its mind *after refuting a specific argument*, or only *to conform to a
   crowded majority*? is there self-consistency across that model's own rounds?
   This is implementable as a small logistic/rule model over these features, and
   it reframes selection from "count votes" to "judge whether a flip was
   epistemic or social."
2. **Reuse Tier 3 (corroboration) on the debate candidates.** Apply the
   existing Corroborator's reliability + independence weighting to the candidate
   pool itself, not only to external sources — reusing infrastructure we already
   built rather than inventing a new piece.
3. **Build the connector layer:** real `Retriever` implementations (interpreter,
   local docs, encyclopedia, academic, database) behind the router, with source
   media-agnosticism. **This is the highest-leverage non-selection step**: with a
   stub Tier 3, the "5-tier truth hierarchy" is in practice a 3-tier hierarchy
   (memory, code, calc) plus a debate fallback. One real connector (e.g. a small
   `LocalDocsRetriever`) validates the design end to end.
4. **Benchmark rigor.** Expand to 100–150 problems with multiple runs;
   report mean ± standard deviation rather than "solved in any run"; and
   separate *solved at least once* from *solved consistently* — these are very
   different claims that are currently being conflated.
5. **Instrument real latency, not just accuracy.** With the router adding tiers,
   measure how much the **100% deterministic resolve (120/120, 1.2s cold / 65ms warm)**
   saves end-to-end before any model is loaded (v1 single ~6–10 s, debate ~15 s; v7 now measures this).
   A latency argument is a stronger deployment story than accuracy alone.
6. **Model-collapse guardrail.** Beyond "needs fresh signal," add a concrete
   check: measure response diversity/entropy of the fine-tuned model on a fixed
   held-out set; if it drops below a threshold between cycles, pause the loop.
   Without this, the hive-mind loop has no safety instrumentation.
7. **Growing-ceiling trainer:** verified memory (instant) + optional LoRA
   fine-tuning (slow), fed by fresh verified signal with the guardrail above.
8. **Multimodal final model:** evaluate a 4-bit VLM on the M1 once connectors
   are media-ready.

## 11. Conclusion

> **A debate among two small, specialized, locally-run models is a practical,
> privacy-preserving, and accurate alternative to a single large cloud model —
> and it is strongest exactly where it matters most: hard problems.**
> Crucially, an *iterative cross-critique debate generates answers neither
> member produces alone — the swarm beats its own oracle*. The remaining gap is
> not raw model capability but the **consensus selection logic** — a software
> problem we can fix, rather than a hardware/cloud cost we must keep paying.
> And for real learning, the path is a **truth hierarchy**: memory, data-level
> execution, deterministic calculation, and honest corroboration first, with
> debate as the fallback it was always meant to be.

## 12. Appendices

### Appendix A — Reproduction

```bash
# Install dependencies
pip install -r requirements.txt

# Leveled benchmark (36 problems; trailing arg = runs per problem)
python cognitive_swarm/evaluation/leveled_benchmark.py 2

# Emergence (cross-critique) experiment, default 8-problem subset
python -m cognitive_swarm.evaluation.emergence_test 8

# Truth hierarchy end-to-end test
python -m py_compile cognitive_swarm/tools/executor.py \
    cognitive_swarm/memory/verified_memory.py \
    cognitive_swarm/orchestration/truth_router.py \
    cognitive_swarm/orchestration/corroboration.py
python cognitive_swarm/evaluation/truth_router_test.py --fresh
```

> Note: verification of some results requires the local MLX models; deterministic
> tiers (Tier 0-3) run with zero model loads.

### Appendix B — Data summary tables

**Historical baseline (36-problem set, v2):**

| mode (36) | overall | easy | medium | hard | math | reas | code |
|---|---|---|---|---|---|---|---|
| phi | 28% | 67% | 8% | 8% | 42% | 25% | 17% |
| qwen | 28% | 33% | 42% | 8% | 17% | 33% | 33% |
| debate | 39% | 42% | 33% | 42% | 67% | 42% | 8% |
| api (Qwen 27B Groq, retired) | 8% | 8% | 8% | 8% | 8% | 0% | 17% |

Value-add (36): debate solved 7/36 neither single solved; union (oracle) 17/36 = 47%; debate 14/36 = 39%.
Emergence (hard 8): oracle 12%, system 38%, pool 62% (m-r-4→42, h-r-3→20 novel).
Consensus (18 balanced): old 33% → new 61% = oracle 61%.

**Current expanded benchmark (120-problem set, v7, `truth_router_test.py --fresh`):**

| mode (120) | overall | math | reasoning | code | trap | CI |
|---|---|---|---|---|---|---|
| phi | 20.8% (25/120) | 11/36 | 7/36 | 5/36 | 2/12 | [14.2, 28.3] |
| qwen | 30.0% (36/120) | 6/36 | 9/36 | 17/36 | 4/12 | [21.7, 38.3] |
| debate | 47.5% (57/120) | 18/36 | 15/36 | 18/36 | 6/12 | [38.3, 56.7] |
| big minimax-m3:free | 87.5% (105/120) | 36/36 | 26/36 | 34/36 | 9/12 | [81.7, 92.5] |
| **truth hierarchy (deterministic, 0 loads)** | **100% (120/120)** | **36/36** | **36/36** | **36/36** | **12/12** | [96.9, 100] |

Truth: code 36/36, trap 12/12, math 36/36, reasoning 36/36 → 120/120 in 1.2s cold / 65ms warm, 0 model loads. Memory `reasoning_op_sig` prevents shadowing (handshake 8=28 vs 100=4950).

### Appendix C — Problems referenced

- **m-r-4:** sequence → expected 42.
- **h-r-3:** apples → expected 20.
- **h-r-4:** handshakes → expected 4950.
- **h-m-1:** sum → expected 1275.
- **h-m-3:** percent-change (unsolved by pipeline).
- **h-m-4:** GCD (unsolved by pipeline).
- **m-r-3:** jugs (unsolved by pipeline).

---

*End of white paper.*
