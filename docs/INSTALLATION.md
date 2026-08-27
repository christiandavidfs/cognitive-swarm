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

## Development Setup

```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
pytest cognitive_swarm/tests/

# Format code
black cognitive_swarm/
```
