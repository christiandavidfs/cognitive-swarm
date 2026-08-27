"""Simple benchmark for comparing single vs debate modes."""

import time
import json
from pathlib import Path
from typing import List, Dict, Any


class BenchmarkProblem:
    """A single benchmark problem."""
    
    def __init__(self, id: str, question: str, expected_answer: str, category: str = "general"):
        self.id = id
        self.question = question
        self.expected_answer = expected_answer
        self.category = category
    
    def check_answer(self, predicted: str) -> bool:
        """Check if predicted answer matches expected."""
        # Normalize
        pred = predicted.lower().strip()
        exp = self.expected_answer.lower().strip()
        
        # Direct match
        if pred == exp:
            return True
        
        # Check if expected is in predicted
        if exp in pred:
            return True
        
        # Try numeric comparison
        try:
            pred_num = float(pred.replace(",", ""))
            exp_num = float(exp.replace(",", ""))
            return abs(pred_num - exp_num) < 0.01
        except:
            pass
        
        return False


class SimpleBenchmark:
    """Simple benchmark suite."""
    
    def __init__(self):
        self.problems = [
            BenchmarkProblem("math_1", "What is 2+2?", "4", "arithmetic"),
            BenchmarkProblem("math_2", "What is 5+3?", "8", "arithmetic"),
            BenchmarkProblem("math_3", "What is 10-4?", "6", "arithmetic"),
            BenchmarkProblem("math_4", "What is 3*4?", "12", "arithmetic"),
            BenchmarkProblem("math_5", "What is 15/3?", "5", "arithmetic"),
        ]
    
    def run_benchmark(self, orchestrator, mode: str, max_tokens: int = 64) -> Dict[str, Any]:
        """Run benchmark on all problems."""
        results = []
        
        for problem in self.problems:
            start_time = time.time()
            
            try:
                result = orchestrator.run(problem.question, mode=mode)
                correct = problem.check_answer(result.final_answer)
                
                results.append({
                    "id": problem.id,
                    "question": problem.question,
                    "expected": problem.expected_answer,
                    "predicted": result.final_answer,
                    "correct": correct,
                    "latency_ms": result.metrics.get("latency_ms", 0),
                    "confidence": result.confidence,
                })
            except Exception as e:
                results.append({
                    "id": problem.id,
                    "question": problem.question,
                    "expected": problem.expected_answer,
                    "predicted": f"ERROR: {str(e)}",
                    "correct": False,
                    "latency_ms": 0,
                    "confidence": 0,
                })
        
        # Calculate summary
        correct_count = sum(1 for r in results if r["correct"])
        total_count = len(results)
        avg_latency = sum(r["latency_ms"] for r in results) / total_count if total_count > 0 else 0
        
        return {
            "mode": mode,
            "total_problems": total_count,
            "correct": correct_count,
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "avg_latency_ms": avg_latency,
            "results": results,
        }


def print_comparison(single_results: Dict, debate_results: Dict):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON: SINGLE vs DEBATE")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Single':<20} {'Debate':<20}")
    print("-" * 65)
    print(f"{'Accuracy':<25} {single_results['accuracy']:.1%}{'':<15} {debate_results['accuracy']:.1%}")
    print(f"{'Avg Latency (ms)':<25} {single_results['avg_latency_ms']:.0f}{'':<15} {debate_results['avg_latency_ms']:.0f}")
    print(f"{'Correct Answers':<25} {single_results['correct']}/{single_results['total_problems']}{'':<13} {debate_results['correct']}/{debate_results['total_problems']}")
    
    print("\n" + "-" * 65)
    print("DETAILED RESULTS:")
    print("-" * 65)
    
    for s, d in zip(single_results["results"], debate_results["results"]):
        status = "✓" if s["correct"] else "✗"
        debate_status = "✓" if d["correct"] else "✗"
        print(f"\n{s['question']}")
        print(f"  Expected: {s['expected']}")
        print(f"  Single:   {s['predicted'][:50]} [{status}]")
        print(f"  Debate:   {d['predicted'][:50]} [{debate_status}]")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from cognitive_swarm.orchestration.workflow import SwarmOrchestrator, SwarmConfig
    
    # Create orchestrator with real model
    config = SwarmConfig(
        model_path="mlx-community/phi-3.5-mini-instruct-4bit",
        max_rounds=1,
        max_tokens=128,
    )
    
    print("Initializing swarm...")
    orchestrator = SwarmOrchestrator(config)
    
    benchmark = SimpleBenchmark()
    
    print("\nRunning SINGLE mode benchmark...")
    single_results = benchmark.run_benchmark(orchestrator, "single")
    
    print("\nRunning DEBATE mode benchmark...")
    debate_results = benchmark.run_benchmark(orchestrator, "debate")
    
    print_comparison(single_results, debate_results)
    
    # Save results
    output_file = Path("benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "single": single_results,
            "debate": debate_results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    orchestrator.unload()
