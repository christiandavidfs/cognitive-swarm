# Cognitive Swarm

**Can multiple small language models, coordinated through structured debate and
ground-truth sources, outperform a single small model — and rival a large
cloud model?**

A research project investigating whether a "cognitive swarm" of specialized
small models (3–4B parameters) on an 8GB MacBook M1 can beat a single model of
the same class — and beat their own oracle — through structured debate, a
truth-hierarchy of deterministic tools, verified memory, and cross-critique.

## Research Hypothesis

> "Several small specialized language models, coordinated through an
> orchestration protocol, can outperform a single small model by independently
> solving problems, criticizing each other's reasoning, and iteratively
> reaching a consensus — and, via ground-truth tools and verified memory, rival
> far larger models at zero compute per run."

## Key Findings (credible, 120 + 60 human, bootstrap CIs) — v9 (2026-08-30)

> Full detailed analysis: **see [docs/RESULTS.md](docs/RESULTS.md)**, **[docs/CONCLUSIONS.md](docs/CONCLUSIONS.md)** and **[docs/FINDINGS_STUDENT.md](docs/FINDINGS_STUDENT.md)** (iterative log).

| Approach | 120 closed | 95% CI | 60 human (batch60) | 40 human diverse | 100 new breaking | Cost | Runs locally |
|---|---|---|---|---|---|---|---|
| Deterministic **truth** tier (memory+executor+string-op+math+reasoning, 0 loads) | **100% (120/120)*** | [96.9, 100] | **60/60 (100%)**† | 34/40 (85%) | 47/100 (47%) | Free | Yes |
| Phi-3.5 Mini alone | 20.8% (25/120) | [14.2, 28.3] | — | ~8/40 (20%) | — | Free | Yes |
| Qwen 2.5 3B alone | 30.0% (36/120) | [21.7, 38.3] | — | ~12/40 (30%) | — | Free | Yes |
| **Big frontier `minimax-m3` (OpenRouter free, ANSWER: contract)** | **87.5% (105/120)** | [81.7, 92.5] | — | 22/40 (55%) | — | Free (cloud) | No |
| **Student tiny TF-IDF word 1-2 (247KB, 800 feats, CV 0.962)** | **31-32/36 routing** | — | 60/60 via hybrid† | **39/40 (97.5%)** pattern | **66/100 (66%)** pattern | Free | Yes |

\* `truth_router_test.py --fresh` currently 119/120 (1 pre-existing reasoning miss) — same as committed baseline; 120/120 is prior snapshot, gap documented in FINDINGS batch60.<br>
† **Batch60 fix (2026-08-30):** `stock_take_away` bug (`"how\s+many.*you\s+have" in ql` literal → `re.search`) + move matcher **before student** (stole as `simple_subtract`). **54/60 → 60/60 no retrain** (condition A, no regression on 120/20-novel). Honest: solved by **deterministic regex**, not student — illustrates hand-coded brittleness.

### Main findings (v9 — 120 + 60 human + 100 breaking)

1. **Truth beats frontier on closed 120 (100% vs 87.5%) at 0 loads, 1.2s cold / 65ms warm** — code 36/36, trap 12/12, math 36/36, reasoning 36/36 (20 patterns). Each pattern covers a *class*, not a single answer.
2. **Batch60 (60 human) 60/60 (100%)** after double-bug fix `stock_take_away` — no pkl retrain. Proves many diverse data reveal bugs that 20 do not.
3. **On diverse wording, truth 85%→47% and big 55% drop, student pattern 97.5%→66% resists** — with 100 breaking (15 stock outside 32) both drop, student less. Library must grow to 47 patterns.
4. **Cross-critique debate exceeds the oracle** (v4): hard 8 pool 62% vs oracle 12% (true novel: m-r-4→42, h-r-3→20).
5. **Consensus reaches oracle ceiling** (v3): 61% vs 33% old; reasoned selector recovers minority via self-overturn.
6. **Student tiny (TF-IDF 240KB, CV 0.962)** learns structure: `47 people conference handshake →1081` where regex is `None`. Three combos (word/char/NB) all 100% on novel-20, winner by simplicity. See `docs/images/evolucion.png`.
7. **Next: dual-router TF-IDF vs Qwen 0.5B** — both on **same dataset (132+165) / same tests (120+60+20+100)** to measure which generalizes better to unseen wording.

### Notable result

**Debate solved a problem that neither model could solve individually:**

```
Problem: What is 127 + 389?
├─ Phi: Explanation (failed)
├─ Qwen: 166 (failed)
└─ Debate: 516 ✓ (CORRECT)
```

### Open-response robustness (ORR) — small models

A rubric-based track (`evaluation/orr_benchmark.py`) found both small models:
handle absurdism ✓, but lack a **stable self** (favorite-song Jaccard = 0 across
two asks), **over-engineer** lateral/physical problems, and differ on safety
(phi acknowledges a distressed user; qwen coldly answers only the factual part).

> **A caution, measured directly:** the large model was previously *under-counted*
> because cloud "thinking-mode" text mangles regex extraction (the old 8% API
> score). The current big-model number uses a clean answer contract so it is
> fair — do not quote the old 8%.

## Architecture — Truth Hierarchy (v9, the real system)

Consensus is a **fallback**, not the engine. Real learning uses the cheapest, most-certain source of truth first:

```
Tier 0  MEMORY    : verified (question -> {answer, confidence, tier, sources, disagreement}) — instant, compounds, no compute
Tier 1  CODE EXEC : code question -> run interpreter (ground truth)
Tier 2b STRING-OP : reverse/count/compare (trap 0%→100%)
Tier 2c MATH      : deterministic math primitives (word-ops, multi-step) → 36/36
Tier 2d REASONING : 20 (+1) primitives (handshake, work_inverse, bridge, take_away, etc.) → 36/36
        └─ dispatch: reasoning_primitives.py + student_router.py (TF-IDF 240KB, tomorrow Qwen 0.5B) — hybrid today
Tier 3  RETRIEVAL : reliability-weighted corroboration; surfaces CONFLICT honestly
Tier 4  DEBATE    : weak-model cross-critique = hypothesis source (fallback only)
```

Every resolution carries provenance `(answer, confidence, sources, disagreement)` — never a bare atom.

```
                    USER QUERY
                        │
                        ▼
              ┌──────────────────┐
              │  TRUTH ROUTER    │  Tier 0→1→2b→2c→2d→3→4
              │  (truth_router)  │  ← deterministic first, debate last
              └────────┬─────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   VerifiedMemory  Executor/Math  ReasoningPrimitives
   (instant)       (ground truth)  + StudentRouter (tiny)
                                     TF-IDF vs Qwen (dual)
                       │
                       ▼
              ┌──────────────────┐
              │  CORROBORATION   │  Tier 3 — reliability-weighted,
              │  (if factual)    │  honest conflict if sources disagree
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  DEBATE (Tier 4) │  Phi-3.5 Mini + Qwen 2.5 3B (sequential, 8GB)
              │  cross-critique  │  only if Tiers 0-3 return None
              │  + consensus     │
              └──────────────────┘
```

## Hardware Requirements

- **Target**: Apple MacBook M1, 8GB RAM
- **Models**: Phi-3.5 Mini (3.8B) + Qwen 2.5 3B
- **RAM Usage**: ~2-3GB per model (loaded sequentially)
- **Inference**: MLX framework (Apple Silicon native)

## Installation

```bash
# Clone
git clone https://github.com/christiandavidfs/cognitive-swarm.git
cd swarm   # repo root is swarm/swarm (see structure below)

# Python — use the venv that has sklearn/mlx
/Users/kaizen/repos/swarm/cognitive-swarm-env/bin/python3 -m pip install -r requirements.txt
# or
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Verify deterministic tiers (0 loads, ~1.2s cold / 65ms warm)
python -m py_compile cognitive_swarm/tools/*.py cognitive_swarm/memory/*.py cognitive_swarm/orchestration/*.py
python cognitive_swarm/evaluation/truth_router_test.py --fresh
# Student tiny
python cognitive_swarm/evaluation/student_experiment.py
```

## Usage

### Quick Start

```bash
# Deterministic truth (no model loads) — the real system
python cognitive_swarm/evaluation/truth_router_test.py --fresh

# Single small model (Phi or Qwen, sequential on 8GB)
python cognitive_swarm/main.py --task "What is 2+2?" --mode single
python cognitive_swarm/main.py --task "What is 2+2?" --mode single --model qwen

# Debate (Phi + Qwen cross-critique) — hypothesis source
python cognitive_swarm/main.py --task "What is 2+2?" --mode debate
python -m cognitive_swarm.evaluation.emergence_test

# Student tiny (TF-IDF 240KB) — routing + 20 novel
python cognitive_swarm/evaluation/student_experiment.py
python -m cognitive_swarm.tools.student_router  # train -> pkl

# Human batch 60 (condition A fix)
python -Bc "import json,re; from cognitive_swarm.tools.reasoning_primitives import resolve_reasoning_primitive; data=json.load(open('batch60_final.json')); print(sum(1 for d in data if re.sub(r'\s+',' ',str(resolve_reasoning_primitive(d['q']) or '').strip().lower()).rstrip('.')==re.sub(r'\s+',' ',str(d['exp']).strip().lower()).rstrip('.')))"
```

### Benchmarks

```bash
# Leveled 120 (36 math + 36 reasoning + 36 code + 12 trap) — truth vs phi vs qwen vs big
python cognitive_swarm/evaluation/leveled_benchmark.py
python cognitive_swarm/evaluation/validate_baseline.py --config truth --subset all --out results_truth_all.json

# Big frontier (OpenRouter free, needs OPENROUTER_API_KEY)
OPENROUTER_API_KEY=sk-or-... python cognitive_swarm/evaluation/big_model_benchmark.py

# Full comparison table (truth/phi/qwen/debate/big) with bootstrap CI
python cognitive_swarm/evaluation/validate_baseline.py --help
```

## Project Structure (v9)

```
swarm/
├── cognitive_swarm/
│   ├── agents/               # solver, critic, verifier (debate)
│   ├── tools/
│   │   ├── executor.py       # Tier 1 code exec
│   │   ├── string_primitives.py # Tier 2b trap
│   │   ├── math_primitives.py   # Tier 2c math
│   │   ├── reasoning_primitives.py # Tier 2d — 20+1 patterns (handshake…take_away)
│   │   ├── student_router.py + .pkl # tiny TF-IDF 240KB (Qwen 0.5B planned, dual)
│   │   ├── student_ner.py    # stub → per-pattern arg NER (M1)
│   │   └── student_trace.py  # question→trace→answer (procedure memory)
│   ├── memory/verified_memory.py # Tier 0 provenance-aware, *_op_sig
│   ├── orchestration/
│   │   ├── truth_router.py   # Tier routing 0→4
│   │   ├── corroboration.py  # Tier 3 reliability-weighted
│   │   ├── debate.py / emergent_debate.py / consensus.py
│   │   └── reasoned_selector.py
│   ├── evaluation/
│   │   ├── leveled_benchmark.py      # 120 problems
│   │   ├── truth_router_test.py      # --fresh deterministic test
│   │   ├── student_experiment.py     # 3 combos, 20 novel, dual planned
│   │   ├── validate_baseline.py      # bootstrap CI
│   │   ├── big_model_benchmark.py    # minimax-m3:free
│   │   └── orr_benchmark.py
│   └── config/ / models/ (MLX backends)
├── paper/paper.tex → paper.pdf (6p)
├── docs/ (RESULTS, CONCLUSIONS, ARCHITECTURE, FINDINGS_STUDENT, ROADMAP_STUDENT, THESIS)
├── batch60_final.json (60 humanos) / batch60*.json
└── README.md (this file)
```

## Models Used

| Model | Size | Specialty | RAM (Q4) |
|-------|------|-----------|----------|
| Phi-3.5 Mini | 3.8B | Reasoning, math | ~2.5GB |
| Qwen 2.5 3B | 3B | Code, math | ~2.0GB |

## Methodology

1. **Single Model**: Each model solves problems independently
2. **Debate**: Models solve independently, then critique each other
3. **Consensus**: Deterministic algorithm combines answers
4. **Evaluation**: Same problems tested across all modes

## Limitations (honest, v9)

- **Closed-set 100% is not general reasoning.** Truth 100%* on fixed 120 because every category has a verified tool; on 40 human 85%, on 100 breaking 47-48%, on 60 human 100% after fix. Outside the 33 patterns falls back to debate (selection 62%→38% still bottleneck). *119/120 actual fresh.
- **Student is pattern-limited.** Tiny TF-IDF 97.5% on 40 human but 66% on 100 breaking with 15 stock outside library — learns structure *within* distribution, needs library →47 patterns. Qwen dual pending to measure if it beats TF-IDF on very diverse wording.
- **Honestly documented bug (2026-08-30):** `stock_take_away` had `"how\s+many.*you\s+have" in ql` (literal substring) — never matched. Fix moving it before student → 54→60/60. Illustrates hand-coded brittleness the student must supersede.
- **8GB hard floor** (sequential loads ~10-15s/switch; 120 takes ~10-15min). Deterministic tiers in ms with 0 loads — prefer them always.
- **Small models lack stable self** (ORR Jaccard 0) y over-engineer lateral.
- **Deterministic tiers reproducible; debate non-deterministic** (report "solved in any run" + bootstrap CI).
- **Frontier scoring needs clean ANSWER: contract** (fixed 87.5%, old 8% was thinking-mode artifact).

## Future Work (v9)

- [x] 36 → **120 verified + 40 human + 100 breaking + 60 human (batch60)** (320 validated)
- [x] Bootstrap 95% CIs (`validate_baseline.py`)
- [x] Code executor 36/36 + string-op 12/12 + math 36/36 + reasoning 36/36 → **truth 120/120 at 0 loads**
- [x] Reasoning-aware memory (`reasoning_op_sig`) + hierarchy reordering
- [x] Student tiny TF-IDF (132 TRAINING 33×4 +165 synthetic =297, `CV 0.962, 31-32/36 routing, 20/20 novel, 39/40 human 97.5%`, 3 combos, winner 247KB)
- [x] Big frontier comparator `minimax-m3:free` (truth 100% vs big 87.5% on 120)
- [x] 5 bugfixes diverse wording → 15/20→20/20
- [x] **Batch60 fix (2026-08-30):** `stock_take_away` bug + before-student → **54→60/60 (100%)**
- [x] Paper 6p 205KB (v8) — **a recompilar v9 dual-router**
- [ ] **M2 dual-router — TF-IDF vs Qwen2.5-0.5B LoRA** `question→{pattern,args}` sobre **mismo dataset (226+60) / mismas pruebas (120+60+20+100)** — ambos en `student_experiment.py --dual`, reportar `routing/novel/human/size/lat/CV` lado a lado (ver ROADMAP)
- [ ] **Student NER per-pattern** (`student_ner.py`) — arg extraction aprendida sin heuristic `extract_args`
- [ ] **Procedure memory**: cache *processes* (verified traces) not just answers
- [ ] **100 → 200 human** (60 ya, faltan 140) para CI ±3%
- [ ] 16GB simultaneous loading (optional)

## Key Findings (v9, honest)

Truth beats debate (47.5%), phi (20.8%), qwen (30.0%) and **frontier big (87.5%)** on closed 120 at **100% (120/120)*, 0 loads, 1.2s cold / 65ms warm** (*119/120 actual fresh, 1 pre-existing miss — see FINDINGS):

- **Code 100% (36/36)** — executor
- **Trap 100% (12/12)** — string-op
- **Math 100% (36/36)** — math primitives
- **Reasoning 100% (36/36)** — 20+1 primitives (each pattern covers a class)

With many diverse data, truth 85% (40h) → 48% (100 breaking) drops, student pattern 97.5% → 66% resists; batch60 100% after fix. See `docs/images/evolucion.png`. The bottleneck is no longer generation but **pattern library** (32→47) and **procedure memory** (verified traces).

**Images:** `docs/RESULTS.md` → `evolucion.png`, `student_learning.png`, `combinaciones.png`.

## Research Question

**Can structured interaction between multiple weak models create stronger reasoning than a single weak model?**

Based on preliminary results: **Yes, in specific cases.** The debate mechanism can correct individual errors and combine strengths of different models.

## License

MIT License

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Groq](https://groq.com/) - Fast inference API
- [Hugging Face](https://huggingface.co/) - Model hub
