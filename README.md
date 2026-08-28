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

| Approach | all-120 acc | 95% CI | code | Cost | Runs locally |
|---|---|---|---|---|---|
| Deterministic **truth** tier (memory + executor + calc + string-op, 0 loads) | **44.2%** | [35.0, 53.3] | **100%** | Free | Yes |
| Phi-3.5 Mini alone | 20.8% | [14.2, 28.3] | 14% | Free | Yes |
| Qwen 2.5 3B alone | 30.0% | [21.7, 38.3] | 47% | Free | Yes |
| **Big model, `minimax-m3` (frontier-class, OpenRouter free)** | **87.5%** | [81.7, 92.5] | 94% | Free (cloud) | No |

### Main findings

1. **Ground truth beats consensus.** The deterministic `truth` tier (**100% on
   code** by *executing* it; **12/12 on the adversarial trap items** via
   string-op primitives — all at zero model loads) beats the individual small
   models overall (44.2% > 20.8% / 30.0%). Caveat: the win is concentrated in
   the code + trap domains; on reasoning it is 0/36 and the *overall*
   truth-vs-qwen gap is within CI overlap — the decisive edge is category-level.
2. **A frontier model dominates on closed-answer recall** (87.5%). The swarm is
   *not* competing with a frontier model on exact-answer trivia; its leverage is
   the open, debate, and grounded domain plus zero-cost compounding memory.
3. **Cross-critique debate exceeds the oracle** (v4): on a hard subset the
   candidate pool reached 62% vs oracle 12% — the swarm generates answers no
   single member produces alone.
4. **Consensus now reaches the oracle ceiling** (v3): confidence-weighted
   consensus matched best-of-members (61% vs 33% old heuristic).
5. **Selection fix (reasoned selector)**: an *independent self-overturn*
   correction recovers novel-correct minority answers that majority voting
   discards — closing the biggest remaining gap.

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

## Limitations

- Limited to 8GB RAM (can't load multiple models simultaneously); sequential
  model loading adds ~15–90s per problem on the full 120-problem set.
- Small models score far below a frontier model on **closed-answer** recall
  (phi 21% / qwen 30% vs 87.5%) — the swarm's value is not here.
- The deterministic `truth` tier is **0% on reasoning and adversarial "trap"**
  problems purely because it has no string/derivation micro-tools yet (a cheap,
  known next win).
- Small models lack a stable "self"/memory of preferences (ORR song-consistency
  = 0) — relevant to identity-style open questions.
- Results are non-deterministic on single runs; accuracy is reported as
  "solved in any run" with bootstrap CIs, not as one lucky number.
- Large API / frontier models with verbose "thinking-mode" output must be scored
  with a clean answer contract or they are unfairly under-counted.

## Future Work

- [x] Expand benchmark 36 → **120 verified problems** (math/reasoning/code + 12 trap)
- [x] Add validation harness with **bootstrap 95% CIs** (`evaluation/validate_baseline.py`)
- [x] Add **code executor** → 100% on code (was 8% trusting models)
- [x] Fix consensus SELECTION via **reasoned selector** (self-overturn correction)
- [x] Big-model comparator vs **frontier model** via free OpenRouter (`minimax-m3`)
- [x] **String-op micro-primitives** (`count_char`, `count_vowels`, `count_letters`,
      `reverse`, `compare`) — land `truth` tier's trap **0% → 100% (12/12)** at
      zero model loads, lifting truth overall to **44.2%** (was 34.2%)
- [ ] Run **debate** config on the full 120 for a true small-swarm vs frontier number
- [ ] **Procedure / primitive memory**: cache *reasoning processes*, not just
      answers — verified lemmas, worked step-skeletons from winning debate
      traces, and discovered correction rules (extends `verified_memory` to the
      procedure level; RAG corpus stays outside on disk)
- [ ] More repeats per config and more model families (Gemma, Llama)
- [ ] Test with 16GB+ RAM for simultaneous model loading

## Research Question

**Can structured interaction between multiple weak models create stronger reasoning than a single weak model?**

Based on preliminary results: **Yes, in specific cases.** The debate mechanism can correct individual errors and combine strengths of different models.

## License

MIT License

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Groq](https://groq.com/) - Fast inference API
- [Hugging Face](https://huggingface.co/) - Model hub
