# Usage Guide

## Docker — One-Command Validation (recommended for reviewers)

```bash
# No local Python/MLX install needed for deterministic tiers (truth 120/120, batch60/200, student)
git clone https://github.com/christiandavidfs/cognitive-swarm.git && cd swarm
docker build -t swarm .
docker run --rm swarm
# Expected inside container:
#   truth 120/120 (36/36/36/12) 0 loads, 1.2s cold / 69ms warm
#   batch60 60/60, batch200 200/200
#   student TF-IDF 33/36 (91.7%), Qwen 0.5B LoRA 33/36 (150 iters)

# Re-run any deterministic test inside Docker
docker run --rm swarm python cognitive_swarm/evaluation/truth_router_test.py --fresh
docker run --rm swarm python cognitive_swarm/evaluation/student_experiment.py
```

## Basic Usage (local)

### Single Model Mode

```bash
# Phi model (default) - best for reasoning
python cognitive_swarm/main.py --task "A farmer has 17 sheep. All but 9 die. How many?" --mode single

# Qwen model - best for math/code
python cognitive_swarm/main.py --task "What is 127 + 389?" --mode single --model qwen
```

### Debate Mode

```bash
# Phi + Qwen debate
python cognitive_swarm/main.py --task "If 5 machines make 5 widgets in 5 minutes, how long for 100?" --mode debate
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--task` | Problem to solve | (required) |
| `--mode` | single/debate/verified | single |
| `--model` | phi/qwen | phi |
| `--max-tokens` | Max tokens per response | 128 |
| `--max-rounds` | Max debate rounds | 1 |
| `--verbose` | Show detailed logs | false |

## Running Benchmarks

### Basic Benchmark

```bash
python cognitive_swarm/evaluation/benchmark.py
```

### Specialized Benchmark

```bash
# Tests reasoning, math, and code separately
python cognitive_swarm/evaluation/specialized_benchmark.py
```

### Full Comparison (with API)

```bash
# Requires Groq API key
export GROQ_API_KEY=your_key
python cognitive_swarm/evaluation/full_comparison.py
```

## Example Output

### Single Mode
```
$ python cognitive_swarm/main.py --task "What is 2+2?" --mode single

============================================================
TASK: What is 2+2?
MODE: single
============================================================

FINAL ANSWER: 4

CONFIDENCE: 0.80
ROUNDS: 1
LATENCY: 3.2s
============================================================
```

### Debate Mode
```
$ python cognitive_swarm/main.py --task "127 + 389?" --mode debate

============================================================
TASK: 127 + 389?
MODE: debate
============================================================

PHI ANSWER: 516
QWEN ANSWER: 516
FINAL ANSWER: 516
CONSENSUS: strong

ROUNDS: 2
LATENCY: 12.5s
============================================================
```

## Python API

### Basic Usage

```python
from cognitive_swarm.orchestration.workflow import SwarmOrchestrator, SwarmConfig

# Configure
config = SwarmConfig(
    model_path="mlx-community/phi-3.5-mini-instruct-4bit",
    max_tokens=128,
    max_rounds=1,
)

# Create orchestrator
swarm = SwarmOrchestrator(config)

# Run single mode
result = swarm.run("What is 2+2?", mode="single")
print(f"Answer: {result.final_answer}")

# Run debate mode
result = swarm.run("127 + 389?", mode="debate")
print(f"Answer: {result.final_answer}")
print(f"Consensus: {result.consensus_result.agreement_level}")

# Cleanup
swarm.unload()
```

### Multi-Model Debate

```python
from cognitive_swarm.models.multi_backend import MultiModelBackend
from cognitive_swarm.orchestration.phi_qwen_debate import PhiQwenDebate

# Create debate
debate = PhiQwenDebate(max_rounds=1)

# Run debate
result = debate.run_debate("What is 2+2?")

print(f"Phi answer: {result.phi_answer}")
print(f"Qwen answer: {result.qwen_answer}")
print(f"Final answer: {result.final_answer}")
print(f"Latency: {result.total_latency_ms:.0f}ms")
```

### Using Different Models

```python
from cognitive_swarm.models.multi_backend import MultiModelBackend

backend = MultiModelBackend()

# List available models
print(backend.list_models())

# Use Phi for reasoning
backend.load_model("phi")
result = backend.generate("Solve this logic puzzle...")
backend.unload_model()

# Use Qwen for math
backend.load_model("qwen")
result = backend.generate("Calculate 15 * 24")
backend.unload_model()
```

## Configuration

### config.yaml

```yaml
model:
  default: "mlx-community/phi-3.5-mini-instruct-4bit"
  max_tokens: 128
  temperature: 0.7

agents:
  solver_a:
    strategy: "normal"
  solver_b:
    strategy: "skeptical"

orchestration:
  max_rounds: 2
  min_confidence: 0.6
```

## Performance Tips

1. **Use appropriate mode**:
   - Single: Simple, quick answers
   - Debate: Complex reasoning, error correction

2. **Optimize max_tokens**:
   - Lower (32-64): Faster, direct answers
   - Higher (128-256): More detailed responses

3. **Model selection**:
   - Phi: Logic, reasoning, analysis
   - Qwen: Math, code, calculations

4. **Sequential loading**:
   - Only one model loaded at a time
   - Unload between switches
   - Close other apps to free RAM
