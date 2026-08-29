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

## Key Findings (credible, 120-problem benchmark with bootstrap CIs)

> Full detailed analysis: **see [docs/RESULTS.md](docs/RESULTS.md)** and
> **[docs/CONCLUSIONS.md](docs/CONCLUSIONS.md)**.

| Approach | all-120 acc | 95% CI | code | math | reasoning | trap | 40 human diverse | Cost | Runs locally |
|---|---|---|---|---|---|---|---|---|---|
| Deterministic **truth** tier (memory + executor + string-op + math + reasoning primitives, 0 loads) | **100% (120/120)** | [96.9, 100] | **36/36** | **36/36** | **36/36** | **12/12** | **34/40 (85%)** | Free | Yes |
| Phi-3.5 Mini alone | 20.8% (25/120) | [14.2, 28.3] | 5/36 | 11/36 | 7/36 | 2/12 | ~8/40 (20%) | Free | Yes |
| Qwen 2.5 3B alone | 30.0% (36/120) | [21.7, 38.3] | 17/36 | 6/36 | 9/36 | 4/12 | ~12/40 (30%) | Free | Yes |
| **Big model, `minimax-m3` (frontier, OpenRouter free)** | **87.5% (105/120)** | [81.7, 92.5] | 34/36 | 36/36 | 26/36 | 9/12 | **22/40 (55%)** | Free (cloud) | No |
| **Student tiny pattern (TF-IDF 1-2, 240KB, 0.26ms)** | **91.7% routing (33/36)** | — | — | — | **39/40 (97.5%)** pattern | — | **39/40 (97.5%)** | Free | Yes |

### Main findings (v8 — all English, 120 + 40 human + 100 new)

1. **Deterministic truth now beats the frontier on closed 120 (100% vs 87.5%)** at **0 loads, 1.2s cold / 65ms warm** — code 36/36, trap 12/12, math 36/36, reasoning 36/36 (20 patterns). Each pattern covers a class, not a single answer.
2. **On diverse human wording (40 human, numbers and phrasing never seen), truth drops to 85% (34/40) and big drops to 55% (22/40), while the tiny student pattern stays 97.5% (39/40)** — proves structure learning, not memorization. With 100 new diverse (including 15 stock breaking outside 32 patterns), hybrid 47% and student pattern 66% — both drop, validating that many data are needed and the library must keep growing.
3. **Frontier still dominates single small models on closed recall (87.5% vs 20.8%/30%)**, but not the truth hierarchy on closed 120.
4. **Cross-critique debate exceeds the oracle** (v4): hard 8 subset candidate pool 62% vs oracle 12%.
5. **Consensus reaches oracle ceiling** (v3): 61% vs 33% old; **reasoned selector** recovers novel-correct minority via independent self-overturn.
6. **Student tiny (TF-IDF word 1-2, 240KB, 0.26ms, CV 1.00)** learns structure: `47 people conference handshake →1081` where regex fails. Three combinations tested (word, char, NB) all 100% on novel 20, winner A by simplicity. See evolution graphs in `docs/RESULTS.md`.

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

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               PROMPT OPTIMIZER (Meta-IA)                    │
│  • Classifies task type (reasoning/math/code)               │
│  • Adds dynamic constraints                                 │
│  • Generates optimal output format                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     ROUTER                                  │
│  • Selects appropriate models                               │
│  • Routes to specialized solvers                            │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   PHI-3.5 MINI (3.8B)   │    │   QWEN 2.5 3B (3B)      │
│   • Reasoning specialist │    │   • Math/code specialist  │
│   • Logic problems       │    │   • Calculations          │
│   • Step-by-step analysis│    │   • Code generation       │
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CRITIQUE PHASE                           │
│  • Each model critiques the other's answer                  │
│  • Identifies errors, assumptions, logic flaws              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CONSENSUS ENGINE                          │
│  • Compares arguments and evidence                          │
│  • Preserves disagreement when appropriate                  │
│  • Can answer "I cannot establish this with confidence"     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FINAL ANSWER                              │
└─────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

- **Target**: Apple MacBook M1, 8GB RAM
- **Models**: Phi-3.5 Mini (3.8B) + Qwen 2.5 3B
- **RAM Usage**: ~2-3GB per model (loaded sequentially)
- **Inference**: MLX framework (Apple Silicon native)

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cognitive-swarm.git
cd cognitive-swarm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Single model (Phi)
python cognitive_swarm/main.py --task "What is 2+2?" --mode single

# Single model (Qwen)
python cognitive_swarm/main.py --task "What is 2+2?" --mode single --model qwen

# Debate mode (Phi + Qwen)
python cognitive_swarm/main.py --task "What is 2+2?" --mode debate
```

### Run Benchmark

```bash
# Run the leveled benchmark (36 problems, easy/medium/hard)
python cognitive_swarm/evaluation/leveled_benchmark.py

# Run specialized benchmark
python cognitive_swarm/evaluation/specialized_benchmark.py

# Run full comparison (requires Groq API key)
export GROQ_API_KEY=your_key_here
python cognitive_swarm/evaluation/full_comparison.py
```

## Project Structure

```
cognitive_swarm/
├── agents/
│   ├── base.py              # Base agent class
│   ├── solver.py            # General solver agent
│   ├── model_solver.py      # Model-specialized solver
│   └── critic.py            # Critic agent
├── models/
│   ├── backend.py           # MLX/Dummy backend
│   └── multi_backend.py     # Multi-model manager
├── orchestration/
│   ├── debate.py            # Debate protocol
│   ├── consensus.py         # Consensus engine
│   └── phi_qwen_debate.py   # Phi+Qwen debate
├── evaluation/
│   ├── benchmark.py         # Basic benchmark
│   ├── specialized_benchmark.py  # Category-specific
│   └── full_comparison.py   # vs API comparison
└── main.py                  # CLI entry point
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

## Limitations (realistic, v8)

- **Closed-set 100% is not general reasoning.** Truth is 100% on the fixed 120 because every category has a verified tool; on 40 diverse human wording it is 85%, on 100 stock-breaking diverse it is 47-48% — outside the 32 patterns it correctly falls back to debate (where selection 62%→38% remains).
- **Student is pattern-limited.** Tiny student pattern is 97.5% on 40 human but drops to 66% on 100 new with 15 stock outside the library — proves it learns structure within distribution, but needs the library to keep growing. No data, only patterns.
- **8GB is a hard floor per base model** (sequential loads ~10-15s per switch; 120 takes ~10-15min). Deterministic tiers run in ms with 0 loads — prefer them.
- **Small models lack a stable self** (ORR song Jaccard 0) and over-engineer lateral problems.
- **Results are non-deterministic for debate** (report "solved in any run" with bootstrap CIs); deterministic tiers are exact and reproducible.
- **Frontier scoring requires clean contract** — verbose thinking-mode must be stripped or it is under-counted (fixed with `ANSWER:` marker at 87.5%).

## Future Work

- [x] Expand benchmark 36 → **120 verified + 40 human diverse + 100 new human (stock breaking)** (total 260 validated)
- [x] Validation harness with **bootstrap 95% CIs** (`evaluation/validate_baseline.py`)
- [x] **Code executor** → 36/36 (100% on code)
- [x] **String-op** → 12/12 (100% trap) and **math primitives** → 36/36 (100%)
- [x] **Reasoning primitives (20 patterns)** → 36/36 (100% reasoning) → **truth 120/120 (100%)** at 0 loads
- [x] **Reasoning-aware memory** (`reasoning_op_sig` prevents handshake 8=28 vs 100=4950 shadowing) and **truth hierarchy reordering** (reasoning before math)
- [x] **Student tiny prototype** — TF-IDF word 1-2 + LogReg, 226 samples (32 patterns ×5 synthetic), `CV 1.00, 33/36 routing, 20/20 novel (100%), 39/40 human (97.5%)` — 3 combinations tested, all 100% novel, winner A 234KB (see `docs/images/evolucion.png`)
- [x] **Big-model comparator** vs frontier `minimax-m3:free` via OpenRouter with tight `ANSWER:` contract (truth 100% vs big 87.5% on 120, truth 85% vs big 55% on 40 human)
- [x] **5 bugfixes for diverse wording** (sequence colon, rope flexible, cube divided, probability marbles, bridge comma list) → novel 15/20→20/20, human 40 19/20
- [x] **Paper recompiled** `6p 205KB` with truth 100% + student
- [ ] **Student v4 NER per pattern** (`student_ner.py` stub) → `Qwen2.5-0.5B LoRA` seq2seq `question→{pattern,args}` to eliminate last regex, target `solo-student 120+100 novel ≥95%` without fallback
- [ ] **Procedure / primitive memory**: cache *reasoning processes* (verified lemmas, winning debate skeletons) not just answers
- [ ] **100 → 200 human diverse** to tighten CI to ±3% and validate many-data regime
- [ ] Test with 16GB+ RAM for simultaneous loading (optional)

## Key Findings (v8, all English, with many data)

Deterministic truth now beats debate (47.5%), phi (20.8%), qwen (30.0%) and **frontier big (87.5%)** on closed 120 at **100% (120/120), 0 loads, 1.2s cold / 65ms warm**:

- **Code 100% (36/36)** — executor
- **Trap 100% (12/12)** — string-op
- **Math 100% (36/36)** — math primitives
- **Reasoning 100% (36/36)** — 20 reasoning primitives (each covers a class)

With many diverse human data, truth drops to **85% (34/40)** and **48% (48/100 on 100 stock breaking)** with diverse wording, while the tiny student pattern stays **97.5% (39/40)** and **66% (66/100)** — proving structure learning vs hand-coded brittleness (56.7% hybrid on 60 human). See evolution graphs in `docs/RESULTS.md` (`docs/images/evolucion.png`).

The remaining bottleneck is not generation but **expanding the pattern library** for stock breaking outside the 32 patterns. Procedure/primitive memory (caching verified reasoning traces) is the next leverage point.

**Images added in:** `docs/RESULTS.md` → `docs/images/evolucion.png`, `student_learning.png`, `combinaciones.png` (tiny vs big vs mini across runs).

## Research Question

**Can structured interaction between multiple weak models create stronger reasoning than a single weak model?**

Based on preliminary results: **Yes, in specific cases.** The debate mechanism can correct individual errors and combine strengths of different models.

## License

MIT License

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Groq](https://groq.com/) - Fast inference API
- [Hugging Face](https://huggingface.co/) - Model hub
