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


def _load():
    # LoRA not yet trained — return None so caller falls back to TF-IDF/regex.
    # When trained: load MLX model + LoRA adapter and return pipeline.
    if not MODEL_PATH.exists():
        return None
    try:
        # from mlx_lm import load  # lazy import, only when adapter exists
        # model, tokenizer = load(str(MODEL_PATH))
        # return (model, tokenizer)
        return None  # placeholder until training run
    except Exception:
        return None


def predict(question: str) -> Optional[Tuple[str, float]]:
    """Qwen router predict — scaffold returns None until LoRA trained."""
    payload = _load()
    if payload is None:
        return None
    # When loaded:
    # prompt = f"Question: {question}\nPattern:"
    # out = generate(model, tokenizer, prompt, max_tokens=16)
    # parse JSON {"pattern": "..."} and confidence from logits
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
    """Train Qwen 0.5B LoRA — real training (needs 8GB + mlx_lm.lora).

    Data: same TRAINING_DATA (132) + synthetic (165) =297, format:
        {"prompt": "Question: ...\\nPattern:", "completion": '{"pattern": "handshake", "args": {"n": 47}}'}
    LoRA: r=8, alpha=16, dropout 0.05, 3 epochs, lr 2e-4, batch 4.
    Until run, returns stub so dual harness can report honestly.
    """
    if not Path("cognitive_swarm/tools/student_router.py").exists():
        return {"error": "student_router not found"}
    return {
        "status": "not_trained",
        "note": "Run mlx_lm.lora with 297 samples (33×4 + 33×5) to produce LoRA adapter at qwen_router_lora/",
        "base": BASE_MODEL,
        "lora": "r=8 alpha=16 dropout 0.05 epochs 3 lr 2e-4 batch 4",
        "dataset": "132 TRAINING_DATA + 165 synthetic =297, 33 classes",
        "next": "python -m cognitive_swarm.tools.qwen_lora_finetune --dual",
    }
