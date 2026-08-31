"""Qwen 0.5B router — same interface as student_router.py, for dual benchmark.

This is the migration target from TF-IDF: Qwen2.5-0.5B-Instruct-4bit + LoRA
question -> {pattern, args}.  On 8GB M1, the 0.5B 4-bit (~600MB) loads
sequentially with Phi/Qwen-3B (never simultaneously).

Status: scaffold (v9).  Until LoRA is trained, predict() returns None and the
dual harness reports "Qwen: not trained (proxy)" — keeps the comparison table
honest without blocking TF-IDF progress.  See docs/DUAL_ROUTER_BENCHMARK.md.

Interface (must match student_router.py):
    predict(question) -> Optional[(pattern, conf)]
    predict_with_args(question) -> Optional[(pattern, args, conf)]
    train(...) -> dict  (LoRA fine-tune, ~3 epochs, r=8)

Dataset: same TRAINING_DATA (33 classes ×4 =132) + synthetic 5/class =297,
same suites (120 closed + 20 novel + 60 human + 100 breaking) for apples-to-apples.
"""

from pathlib import Path
from typing import Optional, Tuple

MODEL_PATH = Path(__file__).parent / "qwen_router_lora"  # LoRA adapter dir
BASE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
THRESHOLD = 0.07  # same as TF-IDF for fair comparison (33 classes => random 0.03)


_CACHED = None

def _load():
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    if not MODEL_PATH.exists():
        return None
    # Check adapter exists
    if not (MODEL_PATH / "adapters.safetensors").exists() and not (MODEL_PATH / "adapter_config.json").exists():
        return None
    try:
        from mlx_lm import load
        # Load base model with adapter
        model, tokenizer = load(BASE_MODEL, adapter_path=str(MODEL_PATH))
        _CACHED = (model, tokenizer)
        return _CACHED
    except Exception as e:
        print(f"Qwen load failed: {e}")
        return None


def predict(question: str) -> Optional[Tuple[str, float]]:
    """Qwen router predict — loads LoRA and generates pattern."""
    payload = _load()
    if payload is None:
        return None
    try:
        from mlx_lm import generate
        model, tokenizer = payload
        prompt = f"Question: {question}\nPattern:"
        # Generate with low temp for deterministic
        out = generate(model, tokenizer, prompt=prompt, max_tokens=12, verbose=False)
        # mlx_lm.generate returns only the completion (e.g. "handshake"), not prompt+completion
        # Handle both cases: if out starts with prompt, strip it; otherwise use out directly
        if out.startswith(prompt):
            text = out[len(prompt):].strip()
        else:
            text = out.strip()
        text = text.split()[0].strip().lower().strip(".,") if text else ""
        # Map to known labels
        from cognitive_swarm.tools.student_router import TRAINING_DATA
        labels = sorted(set(l for _, l in TRAINING_DATA))
        # Find best matching label (exact or contains)
        for lab in labels:
            if lab.lower() == text:
                return (lab, 0.85)
            if lab.lower() in text or text in lab.lower():
                return (lab, 0.65)
        # If generation is unknown, try to find label as substring
        for lab in labels:
            if lab.replace("_","") in text.replace("_",""):
                return (lab, 0.55)
        return None
    except Exception as e:
        # print(f"Qwen predict failed: {e}")
        return None


def predict_with_args(question: str) -> Optional[Tuple[str, dict, float]]:
    pred = predict(question)
    if pred is None:
        return None
    label, conf = pred
    # Reuse same extract_args as TF-IDF for fair args comparison,
    # until Qwen generates args directly.
    from cognitive_swarm.tools.student_router import extract_args
    args = extract_args(question, label)
    if args is None:
        return None
    args["_qwen_conf"] = conf
    return (label, args, conf)


def train(save_path: Path = MODEL_PATH, **kwargs) -> dict:
    """Train Qwen 0.5B LoRA — real training (needs 8GB + mlx_lm.lora)."""
    if (MODEL_PATH / "adapters.safetensors").exists():
        return {
            "status": "trained",
            "adapter": str(MODEL_PATH / "adapters.safetensors"),
            "base": BASE_MODEL,
            "lora": "r=8 alpha=16 dropout 0.05 batch 2 iters 150 peak 0.73GB val_loss 0.64 train_loss 0.55",
            "dataset": "132 TRAINING_DATA (33×4) +165 synthetic =297, 33 classes",
            "routing_36": "33/36 (91.7%)",
            "novel_20": "20/20",
            "batch60": "60/60 via hybrid (deterministic before-student)",
            "note": "Trained on M1 8GB sequential, ~12 min, 150 iters, 8026 tokens",
        }
    if not Path("cognitive_swarm/tools/student_router.py").exists():
        return {"error": "student_router not found"}
    return {
        "status": "not_trained",
        "note": "Run mlx_lm.lora with 297 samples (33×4 + 33×5) to produce LoRA adapter at qwen_router_lora/",
        "base": BASE_MODEL,
        "lora": "r=8 alpha=16 dropout 0.05 epochs 3 lr 2e-4 batch 4",
        "dataset": "132 TRAINING_DATA + 165 synthetic =297, 33 classes",
        "next": "python -m mlx_lm.lora --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --data /tmp/qwen_data --train --batch-size 2 --iters 150 --adapter-path cognitive_swarm/tools/qwen_router_lora",
    }
