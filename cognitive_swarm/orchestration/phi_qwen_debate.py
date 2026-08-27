"""Real debate between Phi and Qwen specialized models."""

from dataclasses import dataclass, field
from typing import Optional, List
import time
import logging

from cognitive_swarm.agents.base import AgentContext, AgentResponse
from cognitive_swarm.models.multi_backend import MultiModelBackend

logger = logging.getLogger(__name__)


@dataclass
class DebateRound:
    """A single round of debate."""
    phi_response: Optional[AgentResponse] = None
    qwen_response: Optional[AgentResponse] = None
    phi_critique: Optional[str] = None
    qwen_critique: Optional[str] = None
    phi_revision: Optional[str] = None
    qwen_revision: Optional[str] = None


@dataclass
class DebateResult:
    """Final result of the debate."""
    task: str
    phi_answer: str
    qwen_answer: str
    final_answer: str
    consensus_level: str  # "strong", "partial", "none"
    rounds: List[DebateRound]
    total_latency_ms: float
    metadata: dict = field(default_factory=dict)


class PhiQwenDebate:
    """Real debate between Phi (reasoning) and Qwen (math/code).
    
    Protocol:
    1. Both solve independently
    2. Phi critiques Qwen's answer
    3. Qwen critiques Phi's answer
    4. Both revise based on critiques
    5. Consensus or preserve disagreement
    """
    
    def __init__(self, max_rounds: int = 1):
        self.backend = MultiModelBackend()
        self.max_rounds = max_rounds
    
    def run_debate(self, task: str, max_tokens: int = 64) -> DebateResult:
        """Run a full debate session."""
        start_time = time.time()
        rounds = []
        
        # Round 1: Independent solutions
        logger.info("Round 1: Independent solutions")
        round1 = DebateRound()
        
        # Phi solves (reasoning specialist)
        self.backend.load_model("phi", max_tokens=max_tokens)
        try:
            phi_prompt = f"""Think through this problem step by step using logical reasoning.

Problem: {task}

Provide your answer on the last line as:
ANSWER: [your answer]"""
            phi_result = self.backend.generate(phi_prompt)
            round1.phi_response = AgentResponse(
                agent_name="phi",
                content=phi_result.text,
                confidence=0.7,
                metadata={"extracted_answer": self._extract_answer(phi_result.text)},
            )
        finally:
            self.backend.unload_model()
        
        # Qwen solves (math/code specialist)
        self.backend.load_model("qwen", max_tokens=max_tokens)
        try:
            qwen_prompt = self._build_qwen_prompt(task)
            qwen_result = self.backend.generate(qwen_prompt)
            round1.qwen_response = AgentResponse(
                agent_name="qwen",
                content=qwen_result.text,
                confidence=0.7,
                metadata={"extracted_answer": self._extract_answer(qwen_result.text)},
            )
        finally:
            self.backend.unload_model()
        
        rounds.append(round1)
        
        # Round 2: Critique each other
        logger.info("Round 2: Mutual critique")
        
        # Phi critiques Qwen
        self.backend.load_model("phi", max_tokens=max_tokens)
        try:
            phi_critique_prompt = f"""You are a logical reasoning expert. Critique this answer.

Problem: {task}
Qwen's answer: {round1.qwen_response.metadata.get('extracted_answer', 'N/A')}
Qwen's reasoning: {round1.qwen_response.content[:200]}

Is this answer logically correct? Point out any errors. Be concise."""
            phi_critique = self.backend.generate(phi_critique_prompt)
            round1.phi_critique = phi_critique.text
        finally:
            self.backend.unload_model()
        
        # Qwen critiques Phi
        self.backend.load_model("qwen", max_tokens=max_tokens)
        try:
            qwen_critique_prompt = f"""You are a math/code expert. Critique this answer.

Problem: {task}
Phi's answer: {round1.phi_response.metadata.get('extracted_answer', 'N/A')}
Phi's reasoning: {round1.phi_response.content[:200]}

Is this answer mathematically correct? Point out any errors. Be concise."""
            qwen_critique = self.backend.generate(qwen_critique_prompt)
            round1.qwen_critique = qwen_critique.text
        finally:
            self.backend.unload_model()
        
        # Round 3: Revision based on critique
        if self.max_rounds > 1:
            logger.info("Round 3: Revision")
            
            # Phi revises
            self.backend.load_model("phi", max_tokens=max_tokens)
            try:
                phi_revision_prompt = f"""Problem: {task}

Your previous answer: {round1.phi_response.metadata.get('extracted_answer', 'N/A')}
Critique received: {round1.qwen_critique[:200] if round1.qwen_critique else 'None'}

Based on the critique, revise your answer if needed. Be concise.
ANSWER: [your final answer]"""
                phi_revision = self.backend.generate(phi_revision_prompt)
                round1.phi_revision = self._extract_answer(phi_revision.text)
            finally:
                self.backend.unload_model()
            
            # Qwen revises
            self.backend.load_model("qwen", max_tokens=max_tokens)
            try:
                qwen_revision_prompt = f"""Problem: {task}

Your previous answer: {round1.qwen_response.metadata.get('extracted_answer', 'N/A')}
Critique received: {round1.phi_critique[:200] if round1.phi_critique else 'None'}

Based on the critique, revise your answer if needed. Be concise.
ANSWER: [your final answer]"""
                qwen_revision = self.backend.generate(qwen_revision_prompt)
                round1.qwen_revision = self._extract_answer(qwen_revision.text)
            finally:
                self.backend.unload_model()
        
        # Determine final answer
        phi_final = round1.phi_revision or round1.phi_response.metadata.get("extracted_answer", "")
        qwen_final = round1.qwen_revision or round1.qwen_response.metadata.get("extracted_answer", "")
        
        # Simple consensus: if answers match, use it; otherwise use majority
        if phi_final.lower().strip() == qwen_final.lower().strip():
            final_answer = phi_final
            consensus = "strong"
        else:
            # For now, prefer Qwen for math/code, Phi for reasoning
            task_lower = task.lower()
            if any(kw in task_lower for kw in ["calculate", "math", "code", "print", "+", "-", "*", "/"]):
                final_answer = qwen_final
                consensus = "partial"
            else:
                final_answer = phi_final
                consensus = "partial"
        
        total_latency = (time.time() - start_time) * 1000
        
        return DebateResult(
            task=task,
            phi_answer=phi_final,
            qwen_answer=qwen_final,
            final_answer=final_answer,
            consensus_level=consensus,
            rounds=rounds,
            total_latency_ms=total_latency,
            metadata={
                "phi_confidence": round1.phi_response.confidence,
                "qwen_confidence": round1.qwen_response.confidence,
            },
        )
    
    def _build_qwen_prompt(self, task: str) -> str:
        """Build specialized prompt for Qwen."""
        task_lower = task.lower()
        
        if any(kw in task_lower for kw in ["what is", "calculate", "compute", "+", "-", "*", "/", "cost", "price"]):
            return f"""Calculate this math problem. Give ONLY the numerical answer, no explanation.

Problem: {task}

ANSWER:"""
        
        elif any(kw in task_lower for kw in ["print", "output", "python", "code", "execute", "len(", "range("]):
            return f"""What does this code output? Give ONLY the output, no explanation.

Code: {task}

ANSWER:"""
        
        else:
            return f"""Solve: {task}

Provide a concise answer. End with:
ANSWER: [your answer]"""
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from response."""
        import re
        
        # Try ANSWER: format
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try other patterns
        patterns = [
            r"(?:final answer|answer|result)[:\s]*(.+?)(?:\n|$)",
            r"(?:therefore|thus|so)[,\s]+(.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()
        
        # Return last line
        lines = text.strip().split("\n")
        return lines[-1] if lines else text
