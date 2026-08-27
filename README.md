# Cognitive Swarm

**Can multiple small language models, coordinated through structured debate, outperform a single small model?**

A research project investigating whether a "cognitive swarm" of specialized small models (1B-4B parameters) can achieve better reasoning than individual models through structured debate, critique, and consensus.

## Research Hypothesis

> "Several small specialized language models, coordinated through an orchestration protocol, can sometimes outperform a single small model by independently solving problems, criticizing each other's reasoning, and iteratively reaching a consensus."

## Key Findings (Preliminary)

| Configuration | Accuracy | Latency |
|--------------|----------|---------|
| Phi-3.5 Mini (single) | 29% | ~6s |
| Qwen 2.5 3B (single) | 14% | ~10s |
| **Debate (Phi + Qwen)** | **29%** | ~15s |
| API (Qwen 27B) | 0%* | ~1s |

*API results affected by model "thinking mode" behavior

### Notable Result

**Debate solved a problem that neither model could solve individually:**

```
Problem: What is 127 + 389?
├─ Phi: Explanation (failed)
├─ Qwen: 166 (failed)
└─ Debate: 516 ✓ (CORRECT)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     ROUTER                                  │
│  • Classifies task type (reasoning/math/code)               │
│  • Selects appropriate models                               │
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

- Limited to 8GB RAM (can't load multiple models simultaneously)
- Sequential model loading adds latency
- Small benchmark size (7 problems)
- Code problems still challenging for small models

## Future Work

- [ ] Larger benchmark (50+ problems)
- [ ] Add more models (Gemma, Llama)
- [ ] Implement RAG memory system
- [ ] Test with 16GB+ RAM for simultaneous loading
- [ ] Compare against larger API models (GPT-4, Claude)

## Research Question

**Can structured interaction between multiple weak models create stronger reasoning than a single weak model?**

Based on preliminary results: **Yes, in specific cases.** The debate mechanism can correct individual errors and combine strengths of different models.

## License

MIT License

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Groq](https://groq.com/) - Fast inference API
- [Hugging Face](https://huggingface.co/) - Model hub
