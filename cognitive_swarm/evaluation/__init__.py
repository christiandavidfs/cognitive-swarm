"""Evaluation framework - benchmarks and metrics for comparing modes."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkProblem:
    """A single benchmark problem."""
    id: str
    question: str
    expected_answer: str
    category: str = "general"
    difficulty: str = "easy"
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating one problem in one mode."""
    problem_id: str
    mode: str
    predicted_answer: str
    expected_answer: str
    correct: bool
    confidence: float
    latency_ms: float
    tokens_generated: int
    memory_usage_mb: float
    rounds: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ComparisonTable:
    """Comparison table across modes."""
    modes: List[str]
    problem_ids: List[str]
    results: Dict[str, List[EvaluationResult]]
    summary: Dict[str, Dict[str, Any]]


class BenchmarkSuite:
    """Suite of benchmark problems for evaluation."""

    def __init__(self):
        self.problems: List[BenchmarkProblem] = []
        self._load_default_benchmarks()

    def _load_default_benchmarks(self):
        """Load default benchmark problems."""
        # Simple arithmetic
        self.problems.extend([
            BenchmarkProblem(
                id="math_1",
                question="What is 127 + 389?",
                expected_answer="516",
                category="arithmetic",
                difficulty="easy",
            ),
            BenchmarkProblem(
                id="math_2",
                question="What is 15 * 24?",
                expected_answer="360",
                category="arithmetic",
                difficulty="easy",
            ),
            BenchmarkProblem(
                id="math_3",
                question="What is 144 / 12?",
                expected_answer="12",
                category="arithmetic",
                difficulty="easy",
            ),
            BenchmarkProblem(
                id="math_4",
                question="What is 256 - 178?",
                expected_answer="78",
                category="arithmetic",
                difficulty="easy",
            ),
            BenchmarkProblem(
                id="math_5",
                question="What is 45 * 3 + 15?",
                expected_answer="150",
                category="arithmetic",
                difficulty="easy",
            ),
        ])

        # Simple logic
        self.problems.extend([
            BenchmarkProblem(
                id="logic_1",
                question="If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
                expected_answer="No",
                category="logic",
                difficulty="easy",
            ),
            BenchmarkProblem(
                id="logic_2",
                question="If it is raining, then the ground is wet. The ground is wet. Is it necessarily raining?",
                expected_answer="No",
                category="logic",
                difficulty="easy",
            ),
        ])

        # Simple facts
        self.problems.extend([
            BenchmarkProblem(
                id="fact_1",
                question="What is the capital of France?",
                expected_answer="Paris",
                category="facts",
                difficulty="easy",
            ),
            BenchmarkProblem(
                id="fact_2",
                question="How many continents are there on Earth?",
                expected_answer="7",
                category="facts",
                difficulty="easy",
            ),
            BenchmarkProblem(
                id="fact_3",
                question="What is the largest planet in our solar system?",
                expected_answer="Jupiter",
                category="facts",
                difficulty="easy",
            ),
        ])

    def add_problem(self, problem: BenchmarkProblem):
        """Add a custom problem."""
        self.problems.append(problem)

    def get_problems(self, category: Optional[str] = None) -> List[BenchmarkProblem]:
        """Get problems, optionally filtered by category."""
        if category:
            return [p for p in self.problems if p.category == category]
        return self.problems


class Evaluator:
    """Evaluates swarm performance on benchmarks."""

    def __init__(self, swarm, benchmark_suite: BenchmarkSuite):
        self.swarm = swarm
        self.benchmark = benchmark_suite
        self.results: List[EvaluationResult] = []

    def evaluate_problem(
        self,
        problem: BenchmarkProblem,
        mode: str,
    ) -> EvaluationResult:
        """Evaluate a single problem in a specific mode."""
        start_time = time.time()

        # Run swarm
        result = self.swarm.run(problem.question, mode=mode)

        # Check correctness
        correct = self._check_correctness(
            result.final_answer,
            problem.expected_answer,
        )

        return EvaluationResult(
            problem_id=problem.id,
            mode=mode,
            predicted_answer=result.final_answer,
            expected_answer=problem.expected_answer,
            correct=correct,
            confidence=result.confidence,
            latency_ms=result.metrics.get("latency_ms", 0),
            tokens_generated=result.metrics.get("tokens", 0),
            memory_usage_mb=result.metrics.get("peak_memory_mb", 0),
            rounds=result.debate_result.total_rounds,
            metadata=result.metrics,
        )

    def evaluate_all(
        self,
        modes: List[str],
        max_problems: Optional[int] = None,
    ) -> ComparisonTable:
        """Evaluate all problems across all modes."""
        problems = self.benchmark.get_problems()
        if max_problems:
            problems = problems[:max_problems]

        results = {mode: [] for mode in modes}

        for problem in problems:
            logger.info(f"Evaluating {problem.id}")
            for mode in modes:
                try:
                    result = self.evaluate_problem(problem, mode)
                    results[mode].append(result)
                except Exception as e:
                    logger.error(f"Error evaluating {problem.id} in {mode}: {e}")

        # Calculate summary
        summary = {}
        for mode in modes:
            mode_results = results[mode]
            if mode_results:
                correct = sum(1 for r in mode_results if r.correct)
                summary[mode] = {
                    "accuracy": correct / len(mode_results),
                    "avg_latency_ms": sum(r.latency_ms for r in mode_results) / len(mode_results),
                    "avg_confidence": sum(r.confidence for r in mode_results) / len(mode_results),
                    "avg_rounds": sum(r.rounds for r in mode_results) / len(mode_results),
                    "total_problems": len(mode_results),
                }

        return ComparisonTable(
            modes=modes,
            problem_ids=[p.id for p in problems],
            results=results,
            summary=summary,
        )

    def _check_correctness(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected."""
        # Normalize answers
        predicted_norm = self._normalize_answer(predicted)
        expected_norm = self._normalize_answer(expected)

        # Exact match
        if predicted_norm == expected_norm:
            return True

        # Check if expected is contained in predicted
        if expected_norm in predicted_norm:
            return True

        # Check numeric equivalence
        try:
            pred_num = float(predicted_norm)
            exp_num = float(expected_norm)
            return abs(pred_num - exp_num) < 1e-6
        except ValueError:
            pass

        return False

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove common prefixes/suffixes
        answer = answer.lower().strip()
        for prefix in ["the answer is", "answer:", "result:", "therefore"]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        return answer

    def save_results(self, filepath: str):
        """Save evaluation results to file."""
        data = {
            "modes": self.results[0].mode if self.results else [],
            "results": [
                {
                    "problem_id": r.problem_id,
                    "mode": r.mode,
                    "correct": r.correct,
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms,
                }
                for r in self.results
            ],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def print_comparison(self, table: ComparisonTable):
        """Print a formatted comparison table."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        for mode in table.modes:
            if mode in table.summary:
                stats = table.summary[mode]
                print(f"\n{mode.upper()}:")
                print(f"  Accuracy: {stats['accuracy']:.1%}")
                print(f"  Avg Latency: {stats['avg_latency_ms']:.0f}ms")
                print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
                print(f"  Avg Rounds: {stats['avg_rounds']:.1f}")
                print(f"  Problems: {stats['total_problems']}")

        print("\n" + "=" * 80)
