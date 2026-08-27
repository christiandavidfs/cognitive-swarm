"""Compare: Single models vs Debate vs API large model."""

import time
import json
from pathlib import Path
from typing import Dict, Any

# API configuration - using Groq free tier
# Sign up at https://console.groq.com for free API key
import os


class APIModel:
    """Wrapper for API-based large model."""
    
    def __init__(self, api_key: str = None, model: str = "qwen/qwen3.6-27b"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.available = bool(self.api_key)
        
        if not self.available:
            print("Warning: GROQ_API_KEY not set. API comparison disabled.")
            print("Get free key at: https://console.groq.com")
    
    def generate(self, prompt: str, max_tokens: int = 64) -> Dict[str, Any]:
        """Generate using Groq API."""
        if not self.available:
            return {"text": "API not available", "latency_ms": 0}
        
        try:
            from groq import Groq
            
            client = Groq(api_key=self.api_key)
            
            start = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            latency = (time.time() - start) * 1000
            
            return {
                "text": response.choices[0].message.content,
                "latency_ms": latency,
                "tokens": response.usage.total_tokens,
            }
        except Exception as e:
            return {"text": f"API error: {str(e)}", "latency_ms": 0}


def extract_answer(text: str) -> str:
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


def run_full_comparison():
    """Run full comparison: Phi vs Qwen vs Debate vs API."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from cognitive_swarm.agents.model_solver import ModelSolverAgent
    from cognitive_swarm.agents.base import AgentContext
    from cognitive_swarm.models.multi_backend import MultiModelBackend
    from cognitive_swarm.orchestration.phi_qwen_debate import PhiQwenDebate
    
    # Test problems
    problems = [
        # Reasoning
        {"id": "r1", "q": "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "a": "9", "cat": "reasoning"},
        {"id": "r2", "q": "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", "a": "5", "cat": "reasoning"},
        
        # Math
        {"id": "m1", "q": "What is 127 + 389?", "a": "516", "cat": "math"},
        {"id": "m2", "q": "What is 15 * 24?", "a": "360", "cat": "math"},
        {"id": "m3", "q": "If a shirt costs $45 and is 30% off, what is the sale price?", "a": "31.5", "cat": "math"},
        
        # Code
        {"id": "c1", "q": "What does this Python code output: print(2 ** 3 ** 2)?", "a": "512", "cat": "code"},
        {"id": "c2", "q": "In Python, what is len([1, 2, [3, 4]])?", "a": "3", "cat": "code"},
    ]
    
    # Initialize components
    backend = MultiModelBackend()
    phi_solver = ModelSolverAgent("phi", backend, "phi", "reasoning")
    qwen_solver = ModelSolverAgent("qwen", backend, "qwen", "math_code")
    debate = PhiQwenDebate(max_rounds=1)
    api_model = APIModel()
    
    results = {
        "phi": [],
        "qwen": [],
        "debate": [],
        "api": [],
    }
    
    print("\n" + "=" * 70)
    print("FULL COMPARISON: PHI vs QWEN vs DEBATE vs API")
    print("=" * 70)
    
    for problem in problems:
        print(f"\nProblem: {problem['q']}")
        print(f"Expected: {problem['a']}")
        
        # Phi
        context = AgentContext(task=problem["q"])
        phi_resp = phi_solver.run(context, max_tokens=64)
        phi_answer = phi_resp.metadata.get("extracted_answer", "")
        phi_correct = check_answer(phi_answer, problem["a"])
        print(f"  Phi: {phi_answer[:30]} [{'✓' if phi_correct else '✗'}]")
        results["phi"].append({"id": problem["id"], "correct": phi_correct, "answer": phi_answer})
        
        # Qwen
        context = AgentContext(task=problem["q"])
        qwen_resp = qwen_solver.run(context, max_tokens=64)
        qwen_answer = qwen_resp.metadata.get("extracted_answer", "")
        qwen_correct = check_answer(qwen_answer, problem["a"])
        print(f"  Qwen: {qwen_answer[:30]} [{'✓' if qwen_correct else '✗'}]")
        results["qwen"].append({"id": problem["id"], "correct": qwen_correct, "answer": qwen_answer})
        
        # Debate
        debate_result = debate.run_debate(problem["q"], max_tokens=64)
        debate_answer = debate_result.final_answer
        debate_correct = check_answer(debate_answer, problem["a"])
        print(f"  Debate: {debate_answer[:30]} [{'✓' if debate_correct else '✗'}]")
        results["debate"].append({"id": problem["id"], "correct": debate_correct, "answer": debate_answer})
        
        # API
        if api_model.available:
            api_prompt = f"""Solve this problem step by step, then give your final answer.

Problem: {problem['q']}

ANSWER: """
            api_resp = api_model.generate(api_prompt, max_tokens=64)
            api_answer = extract_answer(api_resp["text"])
            api_correct = check_answer(api_answer, problem["a"])
            print(f"  API: {api_answer[:30]} [{'✓' if api_correct else '✗'}] ({api_resp['latency_ms']:.0f}ms)")
            results["api"].append({"id": problem["id"], "correct": api_correct, "answer": api_answer})
        else:
            results["api"].append({"id": problem["id"], "correct": False, "answer": "N/A"})
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for mode in ["phi", "qwen", "debate", "api"]:
        correct = sum(1 for r in results[mode] if r["correct"])
        total = len(results[mode])
        print(f"{mode.upper():<10}: {correct}/{total} ({correct/total:.0%})")
    
    # Per category
    print("\nBy Category:")
    for cat in ["reasoning", "math", "code"]:
        print(f"\n  {cat.upper()}:")
        for mode in ["phi", "qwen", "debate", "api"]:
            cat_results = [r for p, r in zip(problems, results[mode]) if p["cat"] == cat]
            correct = sum(1 for r in cat_results if r["correct"])
            total = len(cat_results)
            if total > 0:
                print(f"    {mode.upper():<10}: {correct}/{total}")
    
    # Save results
    output = {
        "problems": problems,
        "results": results,
        "summary": {
            mode: {
                "correct": sum(1 for r in results[mode] if r["correct"]),
                "total": len(results[mode]),
            }
            for mode in results
        },
    }
    
    with open("comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to comparison_results.json")


if __name__ == "__main__":
    run_full_comparison()
