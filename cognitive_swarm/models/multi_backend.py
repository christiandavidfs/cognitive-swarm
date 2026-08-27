"""Multi-model backend for managing multiple LLMs with sequential loading.

On M1 8GB, we can't load multiple models simultaneously.
This backend loads/unloads models on demand to save RAM.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    path: str
    description: str
    strengths: list  # ["reasoning", "math", "code"]
    ram_estimate_gb: float


class MultiModelBackend:
    """Backend that manages multiple models with sequential loading.
    
    Strategy for M1 8GB:
    - Load one model at a time
    - Unload before switching
    - Cache current model name
    """
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.current_model: Optional[str] = None
        self._backend = None
        self._loaded = False
        
        # Register default models
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default models for M1 8GB."""
        self.register_model(
            name="phi",
            path="mlx-community/phi-3.5-mini-instruct-4bit",
            description="Phi-3.5 Mini - Best for reasoning and math",
            strengths=["reasoning", "math", "logic"],
            ram_estimate_gb=2.5,
        )
        
        self.register_model(
            name="qwen",
            path="mlx-community/Qwen2.5-3B-Instruct-4bit",
            description="Qwen 2.5 3B - Best for code and math",
            strengths=["code", "math", "multilingual"],
            ram_estimate_gb=2.0,
        )
    
    def register_model(self, name: str, path: str, description: str, strengths: list, ram_estimate_gb: float):
        """Register a new model."""
        self.models[name] = ModelInfo(
            name=name,
            path=path,
            description=description,
            strengths=strengths,
            ram_estimate_gb=ram_estimate_gb,
        )
        logger.info(f"Registered model: {name} ({description})")
    
    def list_models(self) -> list:
        """List all registered models."""
        return [
            {
                "name": m.name,
                "description": m.description,
                "strengths": m.strengths,
                "ram_gb": m.ram_estimate_gb,
                "loaded": m.name == self.current_model,
            }
            for m in self.models.values()
        ]
    
    def load_model(self, model_name: str, max_tokens: int = 128, temperature: float = 0.7) -> None:
        """Load a specific model, unloading current if needed."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
        
        # Unload current model if different
        if self.current_model and self.current_model != model_name:
            logger.info(f"Unloading {self.current_model} to load {model_name}")
            self.unload_model()
        
        # Skip if already loaded
        if self.current_model == model_name and self._loaded:
            logger.info(f"Model {model_name} already loaded")
            return
        
        # Import and create MLX backend
        from cognitive_swarm.models.backend import MLXBackend, ModelConfig
        
        model_info = self.models[model_name]
        
        logger.info(f"Loading {model_name}: {model_info.path}")
        logger.info(f"Strengths: {model_info.strengths}")
        logger.info(f"Estimated RAM: {model_info.ram_estimate_gb}GB")
        
        self._backend = MLXBackend()
        config = ModelConfig(
            model_path=model_info.path,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._backend.load_model(config)
        
        self.current_model = model_name
        self._loaded = True
        
        logger.info(f"Model {model_name} loaded successfully")
    
    def unload_model(self) -> None:
        """Unload current model."""
        if self._backend and self._loaded:
            logger.info(f"Unloading {self.current_model}")
            self._backend.unload_model()
            self._backend = None
            self._loaded = False
            self.current_model = None
    
    def generate(self, prompt: str, **kwargs) -> object:
        """Generate using current model."""
        if not self._loaded or not self._backend:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        return self._backend.generate(prompt, **kwargs)
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_memory_usage(self) -> dict:
        """Get memory usage info."""
        if self._backend:
            return self._backend.get_memory_usage()
        return {"rss_mb": 0, "model_loaded": False}
    
    def get_model_for_task(self, task_type: str) -> Optional[str]:
        """Recommend best model for a task type."""
        task_to_strength = {
            "reasoning": "reasoning",
            "logic": "reasoning",
            "math": "math",
            "code": "code",
            "coding": "code",
            "programming": "code",
            "multilingual": "multilingual",
        }
        
        strength = task_to_strength.get(task_type.lower())
        if not strength:
            return None
        
        for model_name, model_info in self.models.items():
            if strength in model_info.strengths:
                return model_name
        
        return None
