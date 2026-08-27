"""Specialized benchmark for comparing Phi vs Qwen vs Debate."""

import time
import json
from pathlib import Path
from typing import List, Dict, Any


class SpecializedProblem:
    """A problem designed to test specific model strengths."""
    
    def __init__(self, id: str, question: str, expected_answer: str, category: str, expected_winner: str = None):
        self.id = id
        self.question = question
        self.expected_answer = expected_answer
        self.category = category  # "reasoning", "math", "code"
        self.expected_winner = expected_winner  # "phi", "qwen", "tie"
    
    def check_answer(self, predicted: str) -> bool:
        """Check if predicted answer matches expected."""
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
            pred_num = float(pred.replace(",", "").replace("$", ""))
            exp_num = float(exp.replace(",", "").replace("$", ""))
            return abs(pred_num - exp_num) < 0.01
        except:
            pass
        
        return False


class SpecializedBenchmark:
    """Benchmark with problems designed to test different model strengths."""
    
    def __init__(self):
        self.problems = [
            # Reasoning problems (Phi should excel)
            SpecializedProblem(
                "reason_1",
                "If all cats are animals, and some animals are black, can we conclude that some cats are black?",
                "No",
                "reasoning",
                "phi",
            ),
            SpecializedProblem(
                "reason_2",
                "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
                "9",
                "reasoning",
                "phi",
            ),
            SpecializedProblem(
                "reason_3",
                "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "5",
                "reasoning",
                "phi",
            ),
            
            # Math problems (Both should do well, Qwen slightly better)
            SpecializedProblem(
                "math_1",
                "What is 127 + 389?",
                "516",
                "math",
                "tie",
            ),
            SpecializedProblem(
                "math_2",
                "What is 15 * 24?",
                "360",
                "math",
                "tie",
            ),
            SpecializedProblem(
                "math_3",
                "If a shirt costs $45 and is on sale for 30% off, what is the sale price?",
                "31.5",
                "math",
                "qwen",
            ),
            
            # Code/logic problems (Qwen should excel)
            SpecializedProblem(
                "code_1",
                "What does this Python code output: print(2 ** 3 ** 2)?",
                "512",
                "code",
                "qwen",
            ),
            SpecializedProblem(
                "code_2",
                "In Python, what is the result of: len([1, 2, [3, 4]])?",
                "3",
                "code",
                "qwen",
            ),
        ]
    
    def run_benchmark(self, solver_fn, mode_name: str, **kwargs) -> Dict[str, Any]:
        """Run benchmark using a solver function.
        
        solver_fn: callable(task: str, **kwargs) -> str (the answer)
        """
        results = []
        
        for problem in self.problems:
            start_time = time.time()
            
            try:
                answer = solver_fn(problem.question, **kwargs)
                correct = problem.check_answer(answer)
                
                results.append({
                    "id": problem.id,
                    "question": problem.question,
                    "expected": problem.expected_answer,
                    "predicted": answer,
                    "correct": correct,
                    "category": problem.category,
                    "expected_winner": problem.expected_winner,
                    "latency_ms": (time.time() - start_time) * 1000,
                })
            except Exception as e:
                results.append({
                    "id": problem.id,
                    "question": problem.question,
                    "expected": problem.expected_answer,
                    "predicted": f"ERROR: {str(e)}",
                    "correct": False,
                    "category": problem.category,
                    "expected_winner": problem.expected_winner,
                    "latency_ms": (time.time() - start_time) * 1000,
                })
        
        # Calculate stats
        correct_count = sum(1 for r in results if r["correct"])
        total_count = len(results)
        avg_latency = sum(r["latency_ms"] for r in results) / total_count if total_count > 0 else 0
        
        # Per-category stats
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"correct": 0, "total": 0}
            categories[cat]["total"] += 1
            if r["correct"]:
                categories[cat]["correct"] += 1
        
        return {
            "mode": mode_name,
            "total_problems": total_count,
            "correct": correct_count,
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "avg_latency_ms": avg_latency,
            "categories": {
                cat: {
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
                for cat, stats in categories.items()
            },
            "results": results,
        }


def print_specialized_comparison(phi_results: Dict, qwen_results: Dict, debate_results: Dict):
    """Print detailed comparison of specialized models."""
    print("\n" + "=" * 80)
    print("SPECIALIZED MODEL COMPARISON: PHI vs QWEN vs DEBATE")
    print("=" * 80)
    
    print("\n--- OVERALL ---")
    print(f"{'Mode':<20} {'Accuracy':<12} {'Avg Latency':<15}")
    print("-" * 50)
    print(f"{'Phi (reasoning)':<20} {phi_results['accuracy']:.1%}{'':<7} {phi_results['avg_latency_ms']:.0f}ms")
    print(f"{'Qwen (math/code)':<20} {qwen_results['accuracy']:.1%}{'':<7} {qwen_results['avg_latency_ms']:.0f}ms")
    print(f"{'Debate Phi+Qwen':<20} {debate_results['accuracy']:.1%}{'':<7} {debate_results['avg_latency_ms']:.0f}ms")
    
    print("\n--- BY CATEGORY ---")
    categories = ["reasoning", "math", "code"]
    for cat in categories:
        phi_acc = phi_results.get("categories", {}).get(cat, {}).get("accuracy", 0)
        qwen_acc = qwen_results.get("categories", {}).get(cat, {}).get("accuracy", 0)
        debate_acc = debate_results.get("categories", {}).get(cat, {}).get("accuracy", 0)
        
        print(f"\n{cat.upper()}:")
        print(f"  Phi:   {phi_acc:.0%}")
        print(f"  Qwen:  {qwen_acc:.0%}")
        print(f"  Debate: {debate_acc:.0%}")
    
    print("\n--- DETAILED RESULTS ---")
    for i, (p, q, d) in enumerate(zip(phi_results["results"], qwen_results["results"], debate_results["results"])):
        p_status = "✓" if p["correct"] else "✗"
        q_status = "✓" if q["correct"] else "✗"
        d_status = "✓" if d["correct"] else "✗"
        
        print(f"\n{p['question']}")
        print(f"  Expected: {p['expected']} [{p['category']}]")
        print(f"  Phi:   {p['predicted'][:40]} [{p_status}]")
        print(f"  Qwen:  {q['predicted'][:40]} [{q_status}]")
        print(f"  Debate: {d['predicted'][:40]} [{d_status}]")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from cognitive_swarm.models.multi_backend import MultiModelBackend
    from cognitive_swarm.agents.model_solver import ModelSolverAgent
    from cognitive_swarm.agents.base import AgentContext
    
    print("Initializing multi-model backend...")
    backend = MultiModelBackend()
    
    # Create specialized solvers
    phi_solver = ModelSolverAgent(
        name="phi_solver",
        backend=backend,
        model_name="phi",
        specialization="reasoning",
    )
    
    qwen_solver = ModelSolverAgent(
        name="qwen_solver",
        backend=backend,
        model_name="qwen",
        specialization="math_code",
    )
    
    benchmark = SpecializedBenchmark()
    
    # Test function for single model
    def single_solver(task, model_name, **kwargs):
        if model_name == "phi":
            solver = phi_solver
        else:
            solver = qwen_solver
        
        context = AgentContext(task=task)
        response = solver.run(context, max_tokens=64)
        return response.metadata.get("extracted_answer", response.content)
    
    # Run benchmarks
    print("\nRunning PHI benchmark...")
    phi_results = benchmark.run_benchmark(
        lambda task, **kw: single_solver(task, "phi", **kw),
        "phi"
    )
    
    print("\nRunning QWEN benchmark...")
    qwen_results = benchmark.run_benchmark(
        lambda task, **kw: single_solver(task, "qwen", **kw),
        "qwen"
    )
    
    # For debate, we'd need to implement the debate protocol with two models
    # For now, use phi as placeholder
    print("\nRunning DEBATE benchmark (placeholder)...")
    debate_results = benchmark.run_benchmark(
        lambda task, **kw: single_solver(task, "phi", **kw),
        "debate"
    )
    
    print_specialized_comparison(phi_results, qwen_results, debate_results)
    
    # Save results
    output_file = Path("specialized_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "phi": phi_results,
            "qwen": qwen_results,
            "debate": debate_results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
