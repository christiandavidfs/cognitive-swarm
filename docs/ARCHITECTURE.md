# Architecture

## System Overview

Cognitive Swarm is a multi-agent reasoning system that coordinates multiple small language models through structured debate to achieve better reasoning than individual models.

## Core Components

### 1. Multi-Model Backend (`MultiModelBackend`)

Manages multiple LLMs with sequential loading to optimize RAM usage on constrained hardware.

```python
class MultiModelBackend:
    models: Dict[str, ModelInfo]  # Registered models
    current_model: Optional[str]  # Currently loaded model
    
    def load_model(name: str)     # Load specific model
    def unload_model()            # Free RAM
    def generate(prompt: str)     # Generate with current model
```

**Key Design Decision**: Models are loaded/unloaded sequentially to stay within 8GB RAM limit.

### 2. Specialized Agents

Each agent is optimized for specific task types:

| Agent | Model | Specialization | Prompt Strategy |
|-------|-------|----------------|-----------------|
| PhiSolver | Phi-3.5 Mini | Reasoning, logic | Step-by-step analysis |
| QwenSolver | Qwen 2.5 3B | Math, code | Direct computation |

### 3. Debate Protocol

```
Round 1: Independent Solutions
├─ Phi solves independently
└─ Qwen solves independently

Round 2: Mutual Critique
├─ Phi critiques Qwen's answer
└─ Qwen critiques Phi's answer

Round 3: Revision (optional)
├─ Phi revises based on Qwen's critique
└─ Qwen revises based on Phi's critique

Consensus:
└─ Combine answers or preserve disagreement
```

### 4. Consensus Engine

Deterministic algorithm (no LLM) that:

1. Groups responses by answer similarity
2. Evaluates evidence quality
3. Preserves disagreement when appropriate
4. Can answer "I cannot establish this with confidence"

```python
class ConsensusEngine:
    def reach_consensus(responses: List[AgentResponse]) -> ConsensusResult:
        # Returns:
        # - final_answer
        # - confidence
        # - agreement_level (strong/partial/none)
        # - supporting_evidence
        # - dissenting_views
```

## Memory Architecture

### Working Memory
- Current task
- Agent responses
- Critiques
- Verification results

### Long-term Memory (Planned)
- SQLite storage
- Local embeddings
- Previous problems and solutions

## Debate Protocol Details

### Round 1: Independent Solution
Each model receives the same problem with its specialized prompt:

**Phi Prompt (Reasoning)**:
```
Think through this problem step by step using logical reasoning.

Problem: {task}

Analyze carefully, then provide your answer on the last line as:
ANSWER: [your answer]
```

**Qwen Prompt (Math/Code)**:
```
Calculate this math problem. Give ONLY the numerical answer, no explanation.

Problem: {task}

ANSWER:
```

### Round 2: Cross-Critique
Each model critiques the other's answer:

**Phi critiques Qwen**:
```
You are a logical reasoning expert. Critique this answer.

Problem: {task}
Qwen's answer: {answer}
Qwen's reasoning: {reasoning}

Is this answer logically correct? Point out any errors. Be concise.
```

**Qwen critiques Phi**:
```
You are a math/code expert. Critique this answer.

Problem: {task}
Phi's answer: {answer}
Phi's reasoning: {reasoning}

Is this answer mathematically correct? Point out any errors. Be concise.
```

### Consensus Decision

```python
if phi_answer == qwen_answer:
    return ConsensusLevel.STRONG
elif is_math_problem(task):
    return qwen_answer  # Prefer Qwen for math
else:
    return phi_answer   # Prefer Phi for reasoning
```

## Hardware Constraints

### M1 8GB Limitations
- **Max simultaneous models**: 1 (or 2x 2B models)
- **RAM per model**: 2-3GB (Q4 quantization)
- **Strategy**: Sequential loading with unload between switches

### Performance Characteristics
| Metric | Value |
|--------|-------|
| Model load time | ~1-2s |
| Inference latency | ~4-8s per query |
| Tokens/second | ~15-25 |
| RAM usage | ~2-3GB per model |

## API Integration

Support for external API comparison via Groq:

```python
class APIModel:
    def __init__(self, api_key: str, model: str = "qwen/qwen3.6-27b"):
        self.client = Groq(api_key=api_key)
    
    def generate(self, prompt: str) -> Dict:
        # Returns: text, latency_ms, tokens
```

## Evaluation Framework

### Benchmark Categories
1. **Reasoning**: Logic puzzles, riddles
2. **Math**: Arithmetic, word problems
3. **Code**: Python output prediction

### Metrics
- Accuracy (correct/total)
- Latency (ms)
- Tokens generated
- RAM usage
- Consensus level
