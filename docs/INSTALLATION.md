# Installation Guide

## Prerequisites

- **Hardware**: Apple MacBook M1/M2/M3 with 8GB+ RAM
- **OS**: macOS 12+
- **Python**: 3.10+

## Quick Install

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

## Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| mlx | Apple ML framework | >=0.4.0 |
| mlx-lm | LLM inference | >=0.0.10 |
| psutil | Memory monitoring | >=5.9.0 |
| pyyaml | Configuration | >=6.0 |
| groq | API comparison | >=0.1.0 |
| lancedb | Local vector store | >=0.8.0 |
| sentence-transformers | Embeddings | >=2.5.0 |
| pyarrow | Columnar format | >=15.0.0 |
| pandas | Data analysis | >=2.0.0 |

### LanceDB Vector Memory (Optional)

LanceDB provides local vector search for the verified memory tier (Tier 0). No external services required — stores data as `.lance` files on disk.

```bash
# Already included in requirements.txt
# Install with: pip install lancedb sentence-transformers pyarrow pandas
```

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dim, ~90MB, CPU-only, downloads on first use)
- **Storage**: `memory/lancedb/` directory (auto-created)
- **Schema**: Hybrid exact-code + vector with metadata filtering (tier, source_type, year, journal, etc.)

## Model Download

Models are automatically downloaded from Hugging Face on first use:

- **Phi-3.5 Mini**: `mlx-community/phi-3.5-mini-instruct-4bit` (~2.5GB)
- **Qwen 2.5 3B**: `mlx-community/Qwen2.5-3B-Instruct-4bit` (~2.0GB)

## API Setup (Optional)

For comparison against larger models:

1. Create free account at [Groq Console](https://console.groq.com)
2. Generate API key
3. Set environment variable:

```bash
export GROQ_API_KEY=gsk_your_key_here
```

## Verify Installation

```bash
# Test basic functionality
python cognitive_swarm/main.py --task "What is 2+2?" --mode single

# Test debate mode
python cognitive_swarm/main.py --task "What is 2+2?" --mode debate

# Test LanceDB memory (optional)
python -c "
from cognitive_swarm.memory.verified_memory_lancedb import VerifiedMemoryLanceDB
mem = VerifiedMemoryLanceDB()
mem.remember('What is 2+2?', '4', tier='calculator')
result = mem.lookup('What is 2+2?')
print('LanceDB memory working:', result['answer'] == '4')
"
```

## Hardware Notes

### M1 8GB
- Can run ONE model at a time (2-3GB RAM)
- Sequential loading for debate mode
- ~15-25 tokens/second

### M1 16GB+
- Can potentially run TWO models simultaneously
- Faster debate (no loading/unloading)
- Better performance

## Troubleshooting

### "No module named mlx_lm"
```bash
pip install mlx mlx-lm
```

### "OutOfMemoryError"
- Only one model can be loaded at a time
- Close other applications
- Use smaller max_tokens

### "API error: 404"
- Check model name in code
- Verify API key is valid
- Check Groq status page

### "No module named pandas" (LanceDB)
```bash
pip install pandas
```

### "No module named lancedb"
```bash
pip install lancedb
```

## Development Setup

```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
pytest cognitive_swarm/tests/

# Run LanceDB memory tests specifically
pytest cognitive_swarm/tests/test_verified_memory_lancedb.py -v

# Format code
black cognitive_swarm/
```
