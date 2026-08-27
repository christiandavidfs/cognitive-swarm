"""Main orchestrator - coordinates all agents and manages workflow."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import time
import os
import sys

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class AgentContext:
    """Context passed between agents."""
    task: str
    history: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    round_number: int = 0
    max_rounds: int = 3


@dataclass
class AgentResponse:
    """Standard response from any agent."""
    agent_name: str
    content: str
    confidence: float = 0.0
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)
    inference_result: object = None


@dataclass
class SwarmConfig:
    """Configuration for the swarm."""
    model_path: str
    max_rounds: int = 3
    max_tokens: int = 512
    temperature: float = 0.7
    min_confidence: float = 0.6
    enable_verification: bool = True
    enable_rag: bool = False


@dataclass
class SwarmResult:
    """Complete result from the swarm."""
    task: str
    final_answer: str
    confidence: float
    debate_result: object
    consensus_result: object
    metrics: Dict[str, Any]
    mode: str


class SwarmOrchestrator:
    """Main orchestrator for the cognitive swarm.

    Modes:
    - single: Single model answer
    - debate: Two solvers + critic
    - verified: Debate + verification
    - full: Complete pipeline with all components
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.backend = None
        self.solvers = []
        self.critic = None
        self.verifier = None
        self.debate_protocol = None
        self.consensus_engine = None

        self._setup_components()

    def _setup_components(self):
        """Initialize all components."""
        from cognitive_swarm.agents.solver import SolverAgent
        from cognitive_swarm.agents.critic import CriticAgent
        from cognitive_swarm.agents.verifier import VerifierAgent
        from cognitive_swarm.orchestration.debate import DebateProtocol
        from cognitive_swarm.orchestration.consensus import ConsensusEngine
        from cognitive_swarm.models.backend import DummyBackend, ModelConfig

        # Check if MLX is available and model_path is not dummy
        mlx_available = False
        if self.config.model_path != "dummy":
            try:
                import mlx_lm
                mlx_available = True
            except ImportError:
                pass

        # Import backend based on availability
        if mlx_available:
            try:
                from cognitive_swarm.models.backend import MLXBackend
                self.backend = MLXBackend()
                logger.info("Using MLX backend")
            except ImportError:
                self.backend = DummyBackend()
                logger.info("Using DummyBackend (MLX import failed)")
        else:
            self.backend = DummyBackend()
            logger.info("Using DummyBackend")

        # Load model
        config = ModelConfig(
            model_path=self.config.model_path,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        self.backend.load_model(config)

        # Create agents - all share the same backend (same model instance)
        self.solvers = [
            SolverAgent("solver_a", self.backend, strategy="normal"),
            SolverAgent("solver_b", self.backend, strategy="skeptical"),
        ]
        self.critic = CriticAgent("critic", self.backend)
        self.verifier = VerifierAgent("verifier", self.backend)

        # Create orchestration components
        self.debate_protocol = DebateProtocol(
            max_rounds=self.config.max_rounds,
            early_stop=True,
        )
        self.consensus_engine = ConsensusEngine(
            min_confidence=self.config.min_confidence,
        )

    def run(self, task: str, mode: str = "debate") -> SwarmResult:
        """Run the swarm on a task."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        logger.info(f"Running in {mode} mode on: {task[:50]}...")

        if mode == "single":
            result = self._run_single(task)
        elif mode == "debate":
            result = self._run_debate(task)
        elif mode == "verified":
            result = self._run_verified(task)
        elif mode == "full":
            result = self._run_full(task)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Calculate metrics
        end_time = time.time()
        end_memory = self._get_memory_usage()

        metrics = {
            "latency_ms": (end_time - start_time) * 1000,
            "peak_memory_mb": max(start_memory, end_memory),
            "memory_delta_mb": end_memory - start_memory,
            "mode": mode,
        }
        result.metrics.update(metrics)

        return result

    def _run_single(self, task: str):
        """Single model answer - no debate."""
        from cognitive_swarm.agents.base import AgentContext

        context = AgentContext(task=task)
        response = self.solvers[0].run(context)

        return SwarmResult(
            task=task,
            final_answer=response.metadata.get("extracted_answer", response.content),
            confidence=response.confidence,
            debate_result=type('DebateResult', (), {
                'rounds': [],
                'final_responses': [response],
                'consensus': None,
                'agreement_reached': True,
                'total_rounds': 1,
            })(),
            consensus_result=type('ConsensusResult', (), {
                'final_answer': response.content,
                'confidence': response.confidence,
                'agreement_level': 'strong',
                'supporting_evidence': [],
                'dissenting_views': [],
                'metadata': {},
            })(),
            metrics={"tokens": response.inference_result.tokens_generated if response.inference_result else 0},
            mode="single",
        )

    def _run_debate(self, task: str):
        """Debate between two solvers with critic."""
        debate_result = self.debate_protocol.run_debate(
            task=task,
            solvers=self.solvers,
            critic=self.critic,
            verifier=None,
        )

        consensus = self.consensus_engine.reach_consensus(debate_result.final_responses)

        return SwarmResult(
            task=task,
            final_answer=consensus.final_answer,
            confidence=consensus.confidence,
            debate_result=debate_result,
            consensus_result=consensus,
            metrics={
                "total_rounds": debate_result.total_rounds,
                "agreement_reached": debate_result.agreement_reached,
            },
            mode="debate",
        )

    def _run_verified(self, task: str):
        """Debate with verification."""
        debate_result = self.debate_protocol.run_debate(
            task=task,
            solvers=self.solvers,
            critic=self.critic,
            verifier=self.verifier,
        )

        # Get verification results
        verification_results = []
        for round_obj in debate_result.rounds:
            for v in round_obj.verifications:
                if hasattr(v, 'metadata') and 'verification_results' in v.metadata:
                    verification_results.extend(v.metadata['verification_results'])

        consensus = self.consensus_engine.reach_consensus(
            debate_result.final_responses,
            verification_results=verification_results,
        )

        return SwarmResult(
            task=task,
            final_answer=consensus.final_answer,
            confidence=consensus.confidence,
            debate_result=debate_result,
            consensus_result=consensus,
            metrics={
                "total_rounds": debate_result.total_rounds,
                "verification_results": len(verification_results),
            },
            mode="verified",
        )

    def _run_full(self, task: str):
        """Full pipeline with all components."""
        return self._run_verified(task)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not HAS_PSUTIL:
            return 0.0
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

    def unload(self):
        """Free all resources."""
        if self.backend:
            self.backend.unload_model()
