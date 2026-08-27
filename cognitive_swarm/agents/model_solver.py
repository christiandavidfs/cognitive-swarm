"""Specialized solver agent that uses a specific model."""

from cognitive_swarm.agents.base import BaseAgent, AgentContext, AgentResponse
from cognitive_swarm.models.multi_backend import MultiModelBackend
import re


class ModelSolverAgent:
    """Solver that uses a specific model for its strengths.
    
    Unlike the regular SolverAgent which uses a shared backend,
    this agent:
    1. Loads a specific model (phi, qwen, etc.)
    2. Uses prompts optimized for that model's strengths
    3. Unloads after use to free RAM
    """
    
    def __init__(
        self,
        name: str,
        backend: MultiModelBackend,
        model_name: str,
        specialization: str,
    ):
        self.name = name
        self.backend = backend
        self.model_name = model_name
        self.specialization = specialization
    
    def run(self, context: AgentContext, **kwargs) -> AgentResponse:
        """Run solver with its specialized model."""
        # Load the specific model
        max_tokens = kwargs.get("max_tokens", 128)
        temperature = kwargs.get("temperature", 0.7)
        
        self.backend.load_model(self.model_name, max_tokens=max_tokens, temperature=temperature)
        
        try:
            # Build specialized prompt
            prompt = self._build_prompt(context)
            
            # Generate
            inference_result = self.backend.generate(prompt, **kwargs)
            
            # Parse response
            response = self._parse_response(inference_result.text)
            
            return AgentResponse(
                agent_name=self.name,
                content=inference_result.text,
                confidence=response["confidence"],
                reasoning=inference_result.text,
                metadata={
                    "model": self.model_name,
                    "specialization": self.specialization,
                    "extracted_answer": response["answer"],
                    "tokens": inference_result.tokens_generated,
                    "latency_ms": inference_result.latency_ms,
                },
                inference_result=inference_result,
            )
        finally:
            # Always unload to free RAM
            self.backend.unload_model()
    
    def _build_prompt(self, context: AgentContext) -> str:
        """Build prompt based on model specialization."""
        
        if self.model_name == "phi":
            # Phi excels at reasoning - ask for step-by-step logic
            return f"""Think through this problem step by step using logical reasoning.

Problem: {context.task}

Analyze carefully, then provide your answer on the last line as:
ANSWER: [your answer]"""
        
        elif self.model_name == "qwen":
            # Qwen excels at math/code - ask for DIRECT computation
            # Detect if it's a math or code problem
            task_lower = context.task.lower()
            
            if any(kw in task_lower for kw in ["what is", "calculate", "compute", "+", "-", "*", "/", "cost", "price"]):
                # Math problem - ask for number only
                return f"""Calculate this math problem. Give ONLY the numerical answer, no explanation.

Problem: {context.task}

ANSWER:"""
            
            elif any(kw in task_lower for kw in ["print", "output", "python", "code", "execute", "len(", "range("]):
                # Code problem - ask for output only
                return f"""What does this code output? Give ONLY the output, no explanation.

Code: {context.task}

ANSWER:"""
            
            else:
                return f"""Solve: {context.task}

Provide a concise answer. End with:
ANSWER: [your answer]"""
        
        else:
            return f"""Solve: {context.task}

ANSWER: [your answer]"""
    
    def _parse_response(self, text: str) -> dict:
        """Parse response to extract answer and confidence."""
        # Try to find ANSWER: format
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            return {
                "answer": match.group(1).strip(),
                "confidence": 0.8,
            }
        
        # Try other patterns
        patterns = [
            r"(?:final answer|answer|result)[:\s]*(.+?)(?:\n|$)",
            r"(?:therefore|thus|so)[,\s]+(.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return {
                    "answer": match.group(1).strip(),
                    "confidence": 0.6,
                }
        
        # Return last line
        lines = text.strip().split("\n")
        last_line = lines[-1] if lines else text
        
        return {
            "answer": last_line,
            "confidence": 0.4,
        }
