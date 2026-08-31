# Cognitive Swarm — reproducible validation (no model download needed for deterministic tiers)
FROM python:3.11-slim

WORKDIR /app

# System deps for MLX fallback + LaTeX (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Verify deterministic tiers (0 model loads) — truth 120/120, batch60/200, student 33/36
RUN python -m py_compile cognitive_swarm/tools/*.py cognitive_swarm/memory/*.py cognitive_swarm/orchestration/*.py && \
    python cognitive_swarm/evaluation/truth_router_test.py --fresh

# Default: show validation help
CMD ["python", "cognitive_swarm/evaluation/truth_router_test.py", "--fresh"]
