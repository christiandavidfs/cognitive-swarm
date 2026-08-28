# Architecture — Cognitive Swarm v5 (truth hierarchy)

> This is the current design blueprint. It supersedes the earlier v1 doc and
> reflects where the project has landed: consensus/debate is a **fallback**, the
> real engine is **ground truth + verified memory + honest corroboration**.

## 1. System overview

A local cognitive swarm on an 8GB MacBook (M1) that coordinates small open
models (Phi-3.5 Mini + Qwen 2.5 3B) to reason at the level of — and for some
problem classes *above* — a single model or a large cloud model.

The single most important architectural principle, discovered empirically:

> **The swarm's value is not "averaging weak models".** It (a) recovers ~100% of
> its combined oracle capability via good consensus, and (b) EXCEEDS the oracle
> when models cross-critique each other's reasoning. Real learning, however,
> does not rely on consensus at all — it uses the **cheapest, most-certain
> source of truth** and a **verified memory**, with consensus only as fallback.

## 2. The truth hierarchy (tier order — most important section)

Real learning answers a question with the *strongest available source of truth*,
in this order (cheapest/most-certain first):

```
Tier 0  MEMORY    : already-verified (question -> {answer, confidence, tier,
                   reliability, sources[], disagreement}). Instant, compounds,
                   no compute. This is accumulating "experience".
Tier 1  CODE EXEC : code-output question -> run the interpreter. Ground truth.
Tier 2  CALC      : self-contained arithmetic -> evaluate deterministically.
Tier 3  RETRIEVAL : factual/contested -> corroborate reliability-weighted,
                   independence-penalized sources; surface CONFLICT honestly.
Tier 4  DEBATE    : weak-model debate = HYPOTHESIS source (fallback only).
```

Routing module: `cognitive_swarm/orchestration/truth_router.py`. It always
returns a structured `Resolution(answer, confidence, sources, disagreement)`,
never a bare atom — so a contested question (history, opinions) can honestly
report "sources disagree" instead of faking certainty.

### Why this matters
- The weak code category went from **8% (via models) to 100% (via the
  interpreter)** — ground truth beats consensus every time.
- Memory turns repetition free: a second pass re-resolves results in **ms with
  zero model loads**.
- Deterministic tiers resolve ~42% of the 36-problem set with **0 model loads**.

## 3. The read/write split (the key insight)

There are TWO fundamentally different kinds of knowledge, and treating them as
one is the classic mistake:

| | IN WEIGHTS (the model) | IN MEMORY / COLLECTION (consumable truth) |
|---|---|---|
| Learn once at train time | Yes | No (read fresh on demand) |
| Updates when source changes | No (stale until retrain) | **Yes — next read is current** |
| Grows model size | Yes | **No — model does not grow** |
| Generalizes (skill) | Strong | Weak |
| Precise & current (facts) | Weak/stale | Strong |
| Checkable against source | No | Yes |
| Failure mode | Hallucinates/stale | Depends on connector quality |

**The property that makes this project interesting:** memory/connectors give you
**up-to-date knowledge with zero model regrowth**. When the source updates, the
next query reads the new answer — no retraining, no bigger model. This is
retrieval-augmented generation / verified memory done right, and it is why the
"ceiling" grows on fixed small hardware.

```python
# Pseudocode
def answer(q):
    if hit := memory.lookup(q):        return hit            # growing, current
    if code := executor.run(q):        return verify(code)   # deterministic truth
    claims = [c for src in connectors for c in src.get(q)]   # live sources
    verdict = corroborate(claims)      # reliability + independence-weighted
    memory.archive(q, verdict)         # so it compounds
    return verdict                      # may be None + conflict report
```

## 4. Source taxonomy & selection policy

**Key reframe: no source is uniformly "good" or "bad".** Even Wikipedia is a
reliable source for stable, cited, enumerable facts (chemical formulas, dates,
physical constants) but a poor authority for contested or fast-moving claims.
Selection is therefore **claim-dependent** and must score each source on four
independent axes:

| Axis | What it measures | Example |
|---|---|---|
| **Authority** | How trustworthy the kind of source is for this claim type | peer-reviewed text > encyclopedia > blog |
| **Proximity / primary vs derivative** | Is it the original, or a copy of a copy? | original paper > summary > reblog |
| **Recency / timeliness** | Is the fact fast-moving or stable? | 2026 source > 2023 for current events; stable facts care less |
| **Independence** | Did it derive from another counted source? | two independent studies > ten citing one |

These feed a per-claim reliability `0..1` + an `independent` boolean. The
**Corroborator** (`orchestration/corroboration.py`) then:
- groups claims by normalized answer,
- weighs each by `reliability + (independence ? bonus : 0)`,
- declares **corroborated** if the winner has a dominant, independent-backed
  weight (a single primary source outweighs ten derived copies),
- otherwise returns **conflict** (no forced single answer) with a disagreement
  report — the honest handling of the "different sources, different versions"
  problem.

### Why "Wikipedia is not a source of truth" is half-right
Right: you must never *trust* it as the sole authority. Half-wrong: you also
shouldn't ban it. You **limit its reliability weight**, mark it `independent`
only when it isn't the shared origin of other sources, and corroborate it
against primary/authoritative sources. The scoring model, not a blanket ban,
is what earns trust.

## 5. The connector / retriever interface

Tier 3's `_collect_source_claims` is implemented behind a pluggable, medium-
agnostic retriever layer (`cognitive_swarm/orchestration/retrievers.py`):

```python
class SourceClaim:      # already in corroboration.py
    source: str
    answer: str
    reliability: float   # 0..1 authority + recency
    independent: bool    # penalize copies
    reason: str = ""

class Retriever(ABC):                    # implemented: TableRetriever,
    name: str                            # LocalDocsRetriever
    categories: Optional[Sequence[TaskType]]  # gate: which question kinds it serves
    def applies_to(self, task_type) -> bool
    def get_claims(self, question: str) -> list[SourceClaim]: ...

MATH_SOURCE     = InterpreterRetriever()   # REPL / CAS (sympy)  -> reliability 1.0
DEFINITIONS     = LocalDocsRetriever(...)  # book collection
ENCYCLOPEDIA    = WebRetriever(...)        # wikipedia/others (low authority weight)
SCHOLARLY       = AcademicDBRetriever(...) # peer-reviewed
STRUCTURED      = DatabaseRetriever(...)   # sql / spreadsheets / unit tests
```

**Gate by category (design decision, "path of least resistance").** Rather than
score every source per query or rely on static reliability alone, each retriever
declares the `TaskType` categories it serves, and the router only queries the
connectors whose gate passes. Factual questions classify as `REASONING` /
`UNKNOWN` and reach the docs + knowledge table; `CODE` and `MATH` never touch a
factual connector, so Tier 3 neither wastes work nor injects irrelevant verbose
claims into corroboration — and it avoids a ~10–15s model load that a
query-by-query source-scoring "Option 3" would have required.

The router's Tier 3 merges every gated retriever's claims and corroborates
them reliability-weighted. Each connector is self-contained — swap a source,
add a source, or have a source update, and knowledge is automatically current
without touching the model.

## 6. The growing-ceiling trainer (hive-mind loop)

Two complementary ways the system's effective ceiling grows on fixed hardware:

```
source of truth + cross-critique debate
        │  (verified (question → answer) labels)
        ▼
verified memory ──→ immediate answers (growing, current, no model growth)
        │
        ▼ (optional, slow)
LoRA fine-tune the small model on verified labels ──→ raises generalization
        │
        ▼
swap the tuned model back into the swarm ──→ next cycle's ceiling is higher
```

- **Memory is the cheap, instant ceiling-raiser**: every verified answer is
  immediately reusable with zero recompute. The **model does not grow** (see
  §3) — this is your "the ceiling grows without growing the model" insight.
- **Fine-tuning is the slow, compounding upgrade** that raises generalization,
  but it requires **fresh verified signal** each cycle (more/updated sources)
  to avoid **model collapse** (the swarm just parrots its own outputs and
  stalls). Fresh connectors = fuel; without them the loop plateaus.

## 7. Multimodal (design target — do not block on it)

The **final model** — the one the swarm eventually trains and that *presents*
answers — ideally should be **multimodal**, and it plays **two roles at once**:

1. **Retrieval-augmented answerer (read-side)** — consumes verified memory +
   connectors at query time. Multimodal lets it consume more source *kinds*:
   images, PDFs, diagrams, charts, tables, screenshots, handwriting.
2. **Fine-tuned student (write-side)** — trained on verified labels to raise its
   own weights (generalization). Multimodal makes the learning signal denser
   (image-caption, vision-question pairs), so each fine-tune cycle raises the
   ceiling more.

### Why multimodal makes the knowledge agent easier

- **Wider connector surface.** The connector layer (section 5) extends from
  text-only to *sensory-corroborated* truth: a chart or scanned table becomes a
  queryable source of truth, not just derived text.
- **Sharper read/write split (section 3).** Vision lets the model *consume a raw
  source* (a figure, a scan) at query time instead of relying on stored
  derived text — the "consume, don't memorize" path made general.
- **Higher fine-tuning ceiling (section 6).** Verified multimodal labels are a
  much denser signal than text alone, so the trainer's slow loop pays off more.

### How to keep it unblocking

- Design every interface to be **source/media-agnostic now**: claims may carry
  a `media` field (path/URI) in addition to text (section 5). Nothing downstream
  branches on modality, so text-only works today and vision slots in later with
  zero rework.
- **Do NOT swap models to multimodal yet.** On 8GB a strong VLM at 4-bit
  (Phi-3.5-vision, Qwen2-VL, LLaVA, InternVL) is heavier than the current
  text pair and contends with memory + connectors, plus the sequential-load
  latency problem (section 10). Keep the text pair as the working swarm; treat
  multimodal as the *target ceiling* of the trainer, not the immediate
  deployment. Wire the Retriever layer media-ready first, and only then
  evaluate a concrete VLM on the M1 (RAM, load time, single-model utility).
- The fine-tune loop already assumes the student and the answerer are the same
  model; making that model multimodal simply raises what both roles can do.

## 8. Consensus & the selection problem (the current bottleneck)

The cross-critique debate *generates* answers above the oracle (hard 8-problem
subset: oracle 12%, candidate pool 62%), but the **selector** returns fewer
(system 38%) because **majority voting throws away novel-correct minority
answers**. Two fixes, in priority order:
1. **Route to ground truth first** (Tier 0-3) so we never need the selector for
   verifiable questions.
2. For genuinely unverifiable questions, replace "majority wins" with a selector
   that trusts a confident, well-argued minority correction (e.g. prefer the
   answer both critique rounds converge on, or that overturns the other's
   stated reasoning).

## 9. Component map

| Module | Role |
|---|---|
| `tools/executor.py` | Sandboxed code interpreter → Tier 1 ground truth |
| `memory/verified_memory.py` | Provenance-aware store; code-aware lookup → Tier 0 |
| `orchestration/truth_router.py` | Tier routing + structured Resolution |
| `orchestration/corroboration.py` | Reliability+independence-weighted agreement, conflict |
| `orchestration/optimized_debate.py` | Consensus-weighted single-pass debate (Tier 4) |
| `orchestration/emergent_debate.py` | Cross-critique debate that beats the oracle |
| `orchestration/confidence_consensus.py` | Confidence/self-consistency voting |
| `models/multi_backend.py` | Sequential model loading (8GB constraint) |

## 10. Hardware & operational constraints

- **M1 8GB**: one model loaded at a time; sequential load-dominant latency
  (~10-15s per model switch). Batch generations while a model is loaded.
- **Deterministic tiers (Tier 0-3) run with ZERO model loads and in ms** —
  prefer them over debate everywhere possible.
- **Small models answer first, then echo/explain** → tight token budget
  (math ~10, code ~24), extract preferring the start of output.
- **Results are non-deterministic** on single runs; report "solved in any run".

## 11. Honest limits (read before extending)

1. **Connector quality is the real bottleneck**, not the model. Memory is only
   as good as the sources + corroboration. Bad connector = plausible-sounding
   wrong knowledge.
2. **Model-collapse risk** for the fine-tune loop: without fresh external signal
   each cycle, the swarm stops improving and may overfit to its own outputs.
3. **8GB is a hard floor per base model.** You can grow memory and fine-tune a
   *small* model, but not a 7B on 4GB RAM. The *system* ceiling grows; the
   neuron ceiling is bounded by the chip.
4. **Consensus selection still discards emergent value** (candidates 62% →
   system 38%) when ground truth isn't available.
5. Some hard reasoning/math is genuinely unsolved by the whole pipeline
   (GCD, percent-change, jugs) — real model limits, not routing.
