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
from cognitive_swarm.orchestration.confidence_consensus import (
    ConfidenceConsensusEngine,
    Vote,
)

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
    4. Collects and combines answers using confidence-weighted voting
    """

    def __init__(self, max_rounds: int = 1, samples_per_model: int = 1):
        self.backend = MultiModelBackend()
        self.optimizer = PromptOptimizer(use_llm=False)
        self.consensus = ConfidenceConsensusEngine()
        self.max_rounds = max_rounds
        self.samples_per_model = samples_per_model

    def run_debate(self, task: str, max_tokens: int = 64) -> OptimizedDebateResult:
        """Run optimized debate with confidence-weighted consensus.

        Uses a category-aware token budget: small models tend to state the answer
        first and then either echo the prompt or explain. Limiting tokens to the
        small cases short-circuits that noise => clean answers, dramatically
        reducing reliance on fragile extraction. Code needs a slightly larger
        budget for multi-line literals.
        """
        start_time = time.time()

        # Step 1: Optimize prompts
        logger.info("Step 1: Optimizing prompts")
        phi_optimized = self.optimizer.optimize(task, model_name="phi")
        qwen_optimized = self.optimizer.optimize(task, model_name="qwen")

        task_type = phi_optimized.task_type.value
        logger.info(f"Task type: {task_type}")

        # Category-aware generation budget (max_tokens is ignored geometry here:
        # we always generate with a small budget; callers may pass a larger LLM
        # budget that we cap per category).
        budget = self._budget_for(task_type, max_tokens)

        votes = []
        phi_tokens = 0
        qwen_tokens = 0

        # Step 2: Sample Phi with a tight token budget -> short clean answer.
        logger.info("Step 2: Running Phi with optimized prompt")
        self.backend.load_model("phi", max_tokens=budget)
        try:
            for i in range(self.samples_per_model):
                res = self.backend.generate(phi_optimized.optimized_prompt)
                answer = self._extract_answer(res.text, phi_optimized.task_type)
                conf = ConfidenceConsensusEngine.extract_reported_confidence(res.text)
                votes.append(Vote(model="phi", answer=answer, reported_confidence=conf,
                                  sample_index=i, is_numeric=self.consensus._is_numeric(answer)))
                phi_tokens += getattr(res, "tokens_generated", 0) or 0
        finally:
            self.backend.unload_model()

        # Step 3: Sample Qwen with the same tight budget.
        logger.info("Step 3: Running Qwen with optimized prompt")
        self.backend.load_model("qwen", max_tokens=budget)
        try:
            for i in range(self.samples_per_model):
                res = self.backend.generate(qwen_optimized.optimized_prompt)
                answer = self._extract_answer(res.text, qwen_optimized.task_type)
                conf = ConfidenceConsensusEngine.extract_reported_confidence(res.text)
                votes.append(Vote(model="qwen", answer=answer, reported_confidence=conf,
                                  sample_index=i, is_numeric=self.consensus._is_numeric(answer)))
                qwen_tokens += getattr(res, "tokens_generated", 0) or 0
        finally:
            self.backend.unload_model()

        # Step 4: Confidence-weighted consensus
        logger.info("Step 4: Reaching confidence-weighted consensus")
        verdict = self.consensus.reach_consensus(
            votes, task_type=phi_optimized.task_type, samples_per_model=self.samples_per_model
        )

        total_latency = (time.time() - start_time) * 1000

        # Build a backward-compatible summary of individual answers.
        phi_ans = next((v.answer for v in votes if v.model == "phi"), "")
        qwen_ans = next((v.answer for v in votes if v.model == "qwen"), "")

        return OptimizedDebateResult(
            task=task,
            phi_answer=phi_ans,
            qwen_answer=qwen_ans,
            final_answer=verdict.final_answer,
            consensus_level=verdict.agreement_level,
            phi_prompt_used=phi_optimized.optimized_prompt,
            qwen_prompt_used=qwen_optimized.optimized_prompt,
            total_latency_ms=total_latency,
            task_type=task_type,
            metadata={
                "phi_tokens": phi_tokens,
                "qwen_tokens": qwen_tokens,
                "confidence": verdict.confidence,
                "cluster_scores": verdict.cluster_scores,
                "votes_per_cluster": verdict.votes_per_cluster,
                "num_votes": verdict.metadata.get("num_votes", 0),
                "winning_models": verdict.metadata.get("winning_models", []),
            },
        )

    def _budget_for(self, task_type: str, requested: int) -> int:
        """Return a tight per-category token budget.

        Small models state the answer first then echo/explain; a small budget
        short-circuits that, yielding clean answers with little parsing. Code
        needs a bit more room for multi-line literals (but not the full 64).
        """
        if task_type == TaskType.CODE.value:
            return max(8, min(requested, 24))
        # math / reasoning / unknown -> very tight.
        return max(4, min(requested, 10))

    def _extract_answer(self, text: str, task_type: Optional[TaskType] = None) -> str:
        """Robustly extract the core answer from a (possibly verbose) response.

        Observed behavior of these small models: they state the ANSWER FIRST and
        then give the reasoning afterwards. So we strongly prefer the START of
        the output over the end:

          1. Isolate the leading "answer block" (text before the first explicit
             reasoning marker / blank-line explanation), then parse it.
          2. Math/numeric: the first standalone number in the block.
          3. Code: a literal (List/Dict) or a multi-line numeric run.
          4. Yes/No boolean answer.
          5. Fall back to explicit markers, then the last non-empty line.
        """
        import re
        if not text:
            return ""
        s = text.strip()

        # Remove confidence lines; never part of the answer.
        s = re.sub(r"(?im)^\s*confidence\s*[:=].*?$", "", s).strip()

        # Leading "answer block": cut at the first reasoning / prompt echo line.
        block = re.split(
            r"(?i)(\n\s*(to solve this|##? *solution|explanation|question:?|"
            r"reasoning|step \d|therefore|the statement|in a marathon|given that|"
            r"the code is))",
            s, maxsplit=1,
        )[0].strip()
        # Also strip leading "# Answer" / "Answer" decoration on its own line.
        block = re.sub(r"(?i)^\s*(#?\s*(answer|output|result|final))\s*[:=]?\s*", "", block).strip()

        # --- Parse the isolated block ---
        if not block:
            block = s

        first_line = block.split("\n")[0].strip()

        # 1) Code: multi-line numeric run starting at the very top, but only
        #    CONSECUTIVE numeric lines (stop at first non-numeric line).
        if task_type == TaskType.CODE:
            lines = block.split("\n")
            run = []
            for ln in lines:
                t = ln.strip()
                if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
                    run.append(t)
                else:
                    break
            if len(run) >= 2:
                return "[" + ", ".join(run) + "]"
            m = re.match(r"(\[[^\]]*\]|\{[^}]*\}|\([^)]*\)|`[^`]*`)", first_line)
            if m:
                return m.group(0).strip("`")

        # 2) First line is a clean standalone number.
        m = re.match(r"^(-?\d+(?:\.\d+)?)\s*$", first_line)
        if m:
            return m.group(1)

        # 3) Binary / short first word.
        m = re.match(r"^(no|yes|true|false)\b", block, re.IGNORECASE)
        if m:
            return m.group(1).lower()

        # 4) First number in the block (math or any block starting with a number).
        m = re.search(r"-?\d+(?:\.\d+)?", block)
        if m and task_type in (TaskType.MATH, TaskType.REASONING):
            return m.group(0)

        # 5) Explicit markers anywhere.
        for marker in ["FINAL PRICE =", "ANSWER =", "NUMBER:", "OUTPUT:", "RESULT:"]:
            if marker in s:
                idx = s.index(marker) + len(marker)
                val = s[idx:].split("\n")[0].strip()
                if val:
                    return val

        # 6) Last non-empty line as a last resort.
        lines = [l for l in block.split("\n") if l.strip()]
        return lines[-1] if lines else block


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
