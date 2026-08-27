"""Model backend abstraction for local LLM inference.

Supports MLX (Apple Silicon native) with future support for llama.cpp.
The abstraction allows swapping inference engines without changing agent code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model backend."""
    model_path: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    context_window: int = 2048


@dataclass
class InferenceResult:
    """Result from model inference."""
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    prompt_tokens: int = 0
    total_tokens: int = 0


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    @abstractmethod
    def load_model(self, config: ModelConfig) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Free model resources."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @abstractmethod
    def get_memory_usage(self) -> dict:
        """Get current memory usage information."""
        pass


class MLXBackend(ModelBackend):
    """MLX backend for Apple Silicon inference.

    Uses mlx-lm for efficient inference on M1/M2/M3 chips.
    Memory-efficient: loads model once, shared across all agents.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config: Optional[ModelConfig] = None
        self._loaded = False
        self._mlx_lm = None

    def load_model(self, config: ModelConfig) -> None:
        """Load model using mlx-lm."""
        try:
            # Defer import to load_model call
            if self._mlx_lm is None:
                import mlx_lm
                self._mlx_lm = mlx_lm
            
            logger.info(f"Loading model: {config.model_path}")
            self.model, self.tokenizer = self._mlx_lm.load(config.model_path)
            self.config = config
            self._loaded = True
            logger.info(f"Model loaded successfully: {config.model_path}")
        except ImportError:
            raise ImportError(
                "mlx-lm not installed. Run: pip install mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text using MLX."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)

        start_time = time.time()

        # Tokenize prompt for counting
        prompt_tokens = len(self.tokenizer.encode(prompt))

        # Create sampler for temperature and top_p
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)

        response = self._mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

        latency_ms = (time.time() - start_time) * 1000
        tokens_generated = len(self.tokenizer.encode(response))
        tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0

        return InferenceResult(
            text=response.strip(),
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens + tokens_generated,
        )

    def unload_model(self) -> None:
        """Free model resources."""
        self.model = None
        self.tokenizer = None
        self._loaded = False
        logger.info("Model unloaded")

    def is_loaded(self) -> bool:
        return self._loaded

    def get_memory_usage(self) -> dict:
        """Get approximate memory usage."""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss_mb": mem_info.rss / (1024 * 1024),
                "vms_mb": mem_info.vms / (1024 * 1024),
                "model_loaded": self._loaded,
                "model_path": self.config.model_path if self.config else None,
            }
        except ImportError:
            return {
                "rss_mb": 0,
                "vms_mb": 0,
                "model_loaded": self._loaded,
                "model_path": self.config.model_path if self.config else None,
            }


class DummyBackend(ModelBackend):
    """Dummy backend for testing without loading a real model.

    Returns predefined responses. Useful for testing orchestration logic.
    """

    def __init__(self, responses: Optional[list] = None):
        self.responses = responses or ["This is a dummy response."]
        self._response_idx = 0
        self._loaded = False
        self.config = None

    def load_model(self, config: ModelConfig) -> None:
        self.config = config
        self._loaded = True

    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        response = self.responses[self._response_idx % len(self.responses)]
        self._response_idx += 1
        return InferenceResult(
            text=response,
            tokens_generated=10,
            latency_ms=1.0,
            tokens_per_second=10.0,
            prompt_tokens=5,
            total_tokens=15,
        )

    def unload_model(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def get_memory_usage(self) -> dict:
        return {"rss_mb": 0, "vms_mb": 0, "model_loaded": self._loaded}
