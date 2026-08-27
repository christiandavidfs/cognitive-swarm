"""Compare original vs optimized prompts."""

import time
import json
from pathlib import Path
from typing import Dict, Any


def check_answer(predicted: str, expected: str) -> bool:
    """Check if answer is correct."""
    pred = predicted.lower().strip()
    exp = expected.lower().strip()
    
    if pred == exp:
        return True
    
    if exp in pred:
        return True
    
    try:
        pred_num = float(pred.replace(",", "").replace("$", ""))
        exp_num = float(exp.replace(",", "").replace("$", ""))
        return abs(pred_num - exp_num) < 0.01
    except:
        pass
    
    return False


def run_comparison():
    """Run comparison of original vs optimized prompts."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from cognitive_swarm.orchestration.optimized_debate import OptimizedDebate
    from cognitive_swarm.orchestration.phi_qwen_debate import PhiQwenDebate
    
    # Test problems
    problems = [
        {"q": "What is 127 + 389?", "a": "516", "cat": "math"},
        {"q": "What is 15 * 24?", "a": "360", "cat": "math"},
        {"q": "If a shirt costs $45 and is 30% off, what is the sale price?", "a": "31.5", "cat": "math"},
        {"q": "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "a": "9", "cat": "reasoning"},
        {"q": "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100?", "a": "5", "cat": "reasoning"},
    ]
    
    # Initialize debates
    original_debate = PhiQwenDebate(max_rounds=1)
    optimized_debate = OptimizedDebate(max_rounds=1)
    
    results = {
        "original": {"correct": 0, "total": 0, "details": []},
        "optimized": {"correct": 0, "total": 0, "details": []},
    }
    
    print("\n" + "=" * 70)
    print("COMPARISON: ORIGINAL vs OPTIMIZED PROMPTS")
    print("=" * 70)
    
    for problem in problems:
        print(f"\nProblem: {problem['q']}")
        print(f"Expected: {problem['a']}")
        
        # Original debate
        orig_result = original_debate.run_debate(problem["q"], max_tokens=64)
        orig_correct = check_answer(orig_result.final_answer, problem["a"])
        print(f"  Original: {orig_result.final_answer[:30]} [{'✓' if orig_correct else '✗'}]")
        
        # Optimized debate
        opt_result = optimized_debate.run_debate(problem["q"], max_tokens=64)
        opt_correct = check_answer(opt_result.final_answer, problem["a"])
        print(f"  Optimized: {opt_result.final_answer[:30]} [{'✓' if opt_correct else '✗'}]")
        
        # Track results
        results["original"]["total"] += 1
        results["optimized"]["total"] += 1
        
        if orig_correct:
            results["original"]["correct"] += 1
        if opt_correct:
            results["optimized"]["correct"] += 1
        
        results["original"]["details"].append({
            "question": problem["q"],
            "answer": orig_result.final_answer,
            "correct": orig_correct,
        })
        results["optimized"]["details"].append({
            "question": problem["q"],
            "answer": opt_result.final_answer,
            "correct": opt_correct,
            "task_type": opt_result.task_type,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    orig_acc = results["original"]["correct"] / results["original"]["total"]
    opt_acc = results["optimized"]["correct"] / results["optimized"]["total"]
    
    print(f"\nOriginal Prompts:  {results['original']['correct']}/{results['original']['total']} ({orig_acc:.0%})")
    print(f"Optimized Prompts: {results['optimized']['correct']}/{results['optimized']['total']} ({opt_acc:.0%})")
    
    improvement = opt_acc - orig_acc
    if improvement > 0:
        print(f"\n✓ IMPROVEMENT: +{improvement:.0%} with optimized prompts")
    elif improvement < 0:
        print(f"\n✗ REGRESSION: {improvement:.0%} with optimized prompts")
    else:
        print(f"\n= NO CHANGE")
    
    # Save results
    with open("prompt_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to prompt_comparison_results.json")


if __name__ == "__main__":
    run_comparison()
