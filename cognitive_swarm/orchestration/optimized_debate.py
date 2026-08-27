"""Enhanced debate with Prompt Optimizer integration."""

from dataclasses import dataclass, field
from typing import Optional, List
import time
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognitive_swarm.agents.base import AgentContext, AgentResponse
from cognitive_swarm.models.multi_backend import MultiModelBackend
from cognitive_swarm.orchestration.prompt_optimizer import PromptOptimizer, TaskType

logger = logging.getLogger(__name__)


@dataclass
class OptimizedDebateResult:
    """Result of optimized debate."""
    task: str
    phi_answer: str
    qwen_answer: str
    final_answer: str
    consensus_level: str
    phi_prompt_used: str
    qwen_prompt_used: str
    total_latency_ms: float
    task_type: str
    metadata: dict = field(default_factory=dict)


class OptimizedDebate:
    """Debate with prompt optimization.
    
    This version:
    1. Analyzes the question type
    2. Generates optimized prompts with constraints
    3. Sends optimized prompts to specialized models
    4. Collects and combines answers
    """
    
    def __init__(self, max_rounds: int = 1):
        self.backend = MultiModelBackend()
        self.optimizer = PromptOptimizer(use_llm=False)
        self.max_rounds = max_rounds
    
    def run_debate(self, task: str, max_tokens: int = 64) -> OptimizedDebateResult:
        """Run optimized debate."""
        start_time = time.time()
        
        # Step 1: Optimize prompts
        logger.info("Step 1: Optimizing prompts")
        phi_optimized = self.optimizer.optimize(task, model_name="phi")
        qwen_optimized = self.optimizer.optimize(task, model_name="qwen")
        
        logger.info(f"Task type: {phi_optimized.task_type.value}")
        logger.info(f"Phi constraints: {phi_optimized.constraints}")
        
        # Step 2: Run Phi with optimized prompt
        logger.info("Step 2: Running Phi with optimized prompt")
        self.backend.load_model("phi", max_tokens=max_tokens)
        try:
            phi_result = self.backend.generate(phi_optimized.optimized_prompt)
            phi_answer = self._extract_answer(phi_result.text)
        finally:
            self.backend.unload_model()
        
        # Step 3: Run Qwen with optimized prompt
        logger.info("Step 3: Running Qwen with optimized prompt")
        self.backend.load_model("qwen", max_tokens=max_tokens)
        try:
            qwen_result = self.backend.generate(qwen_optimized.optimized_prompt)
            qwen_answer = self._extract_answer(qwen_result.text)
        finally:
            self.backend.unload_model()
        
        # Step 4: Consensus
        logger.info("Step 4: Reaching consensus")
        final_answer, consensus = self._reach_consensus(
            task, phi_answer, qwen_answer, phi_optimized.task_type
        )
        
        total_latency = (time.time() - start_time) * 1000
        
        return OptimizedDebateResult(
            task=task,
            phi_answer=phi_answer,
            qwen_answer=qwen_answer,
            final_answer=final_answer,
            consensus_level=consensus,
            phi_prompt_used=phi_optimized.optimized_prompt,
            qwen_prompt_used=qwen_optimized.optimized_prompt,
            total_latency_ms=total_latency,
            task_type=phi_optimized.task_type.value,
            metadata={
                "phi_tokens": phi_result.tokens_generated,
                "qwen_tokens": qwen_result.tokens_generated,
            },
        )
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from response."""
        import re
        
        # Try ANSWER: format
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try NUMBER: format
        match = re.search(r"NUMBER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try OUTPUT: format
        match = re.search(r"OUTPUT:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try other patterns
        patterns = [
            r"(?:final answer|answer|result)[:\s]*(.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()
        
        # Return last non-empty line
        lines = [l for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else text
    
    def _reach_consensus(
        self, 
        task: str, 
        phi_answer: str, 
        qwen_answer: str,
        task_type: TaskType,
    ) -> tuple:
        """Reach consensus between answers."""
        # Normalize answers
        phi_norm = phi_answer.lower().strip()
        qwen_norm = qwen_answer.lower().strip()
        
        # Check if answers match
        if phi_norm == qwen_norm:
            return phi_answer, "strong"
        
        # Check if one contains the other
        if phi_norm in qwen_norm or qwen_norm in phi_norm:
            return phi_answer if len(phi_answer) < len(qwen_answer) else qwen_answer, "partial"
        
        # Task-type based preference
        if task_type == TaskType.MATH:
            # Prefer numeric answers
            if self._is_numeric(phi_answer) and not self._is_numeric(qwen_answer):
                return phi_answer, "partial"
            elif self._is_numeric(qwen_answer) and not self._is_numeric(phi_answer):
                return qwen_answer, "partial"
            elif self._is_numeric(qwen_answer):
                return qwen_answer, "partial"
        
        elif task_type == TaskType.CODE:
            # Prefer shorter, more direct answers
            return qwen_answer if len(qwen_answer) < len(phi_answer) else phi_answer, "partial"
        
        # Default: return both, let user decide
        return f"PHI: {phi_answer} | QWEN: {qwen_answer}", "none"
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text is numeric."""
        try:
            float(text.replace(",", "").replace("$", ""))
            return True
        except:
            return False


# Example usage
if __name__ == "__main__":
    debate = OptimizedDebate()
    
    test_questions = [
        "What is 127 + 389?",
        "What is 15 * 24?",
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        result = debate.run_debate(q, max_tokens=32)
        print(f"Task Type: {result.task_type}")
        print(f"Phi Answer: {result.phi_answer}")
        print(f"Qwen Answer: {result.qwen_answer}")
        print(f"Final Answer: {result.final_answer}")
        print(f"Consensus: {result.consensus_level}")
        print(f"Latency: {result.total_latency_ms:.0f}ms")
