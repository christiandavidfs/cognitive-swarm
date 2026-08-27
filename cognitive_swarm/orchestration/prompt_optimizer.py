"""Prompt Optimizer - Meta-IA that improves prompts before sending to solvers.

This module analyzes questions and generates optimized prompts with:
- Task-type detection (reasoning/math/code)
- Dynamic constraints
- Output format specifications
- Context enrichment
"""

import re
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum


class TaskType(Enum):
    """Task classification."""
    REASONING = "reasoning"
    MATH = "math"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class OptimizedPrompt:
    """Result of prompt optimization."""
    original_question: str
    task_type: TaskType
    optimized_prompt: str
    constraints: List[str]
    output_format: str
    confidence: float
    metadata: Dict = None


class PromptOptimizer:
    """Optimizes prompts by adding constraints and context.
    
    Two modes:
    1. Deterministic: Rule-based optimization (fast, no LLM)
    2. Meta-LLM: Uses a small LLM to optimize prompts (slower, smarter)
    """
    
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self._task_patterns = self._build_task_patterns()
        self._constraint_templates = self._build_constraint_templates()
    
    def _build_task_patterns(self) -> Dict[TaskType, List[str]]:
        """Build patterns for task classification."""
        return {
            TaskType.MATH: [
                r"what is \d+",
                r"calculate",
                r"compute",
                r"solve.*equation",
                r"\d+\s*[\+\-\*\/\^]\s*\d+",
                r"how much",
                r"how many",
                r"cost.*price",
                r"percent.*off",
                r"discount",
                r"total.*sum",
                r"add.*together",
                r"multiply",
                r"divide",
            ],
            TaskType.CODE: [
                r"what does.*print",
                r"what is the output",
                r"python.*code",
                r"execute",
                r"run.*code",
                r"len\(",
                r"range\(",
                r"def ",
                r"class ",
                r"import ",
                r"return ",
                r"for.*in.*:",
                r"if.*else",
                r"while ",
            ],
            TaskType.REASONING: [
                r"if.*then",
                r"can we conclude",
                r"does it follow",
                r"logical",
                r"reason",
                r"why",
                r"explain",
                r"all but",
                r"fewer.*than",
                r"more than",
                r"at least",
                r"at most",
                r"must be",
                r"could be",
                r"possible",
                r"necessary",
                r"sufficient",
            ],
        }
    
    def _build_constraint_templates(self) -> Dict[TaskType, Dict]:
        """Build constraint templates for each task type."""
        return {
            TaskType.MATH: {
                "constraints": [
                    "GIVE ONLY THE NUMERICAL ANSWER",
                    "NO explanation, NO steps, NO units",
                    "If decimal, use point not comma",
                    "If integer, no decimal point",
                    "For percent: DISCOUNT means subtract from original",
                ],
                "output_format": "NUMBER",
                "examples": [
                    "What is 2+2? → 4",
                    "What is 10/3? → 3.333",
                    "45 with 30% discount → 31.5",
                ],
            },
            TaskType.CODE: {
                "constraints": [
                    "GIVE ONLY THE OUTPUT",
                    "NO explanation, NO code, NO quotes",
                    "If error, write ERROR: reason",
                    "If multiple lines, use newline",
                ],
                "output_format": "OUTPUT",
                "examples": [
                    "print(2+2) → 4",
                    "print([1,2,3][1]) → 2",
                ],
            },
            TaskType.REASONING: {
                "constraints": [
                    "Think step by step",
                    "Be concise (max 3 steps)",
                    "End with ANSWER: [answer]",
                    "Answer must be short (1-5 words)",
                ],
                "output_format": "ANSWER: [your answer]",
                "examples": [
                    "If all A are B, and some B are C → Answer: No",
                ],
            },
        }
    
    def optimize(self, question: str, model_name: str = None) -> OptimizedPrompt:
        """Optimize a question into a well-constrained prompt."""
        # Classify task type
        task_type = self._classify_task(question)
        
        # Get constraints for this task type
        template = self._constraint_templates.get(task_type, self._constraint_templates[TaskType.REASONING])
        
        # Build optimized prompt
        if self.use_llm:
            optimized = self._optimize_with_llm(question, task_type, model_name)
        else:
            optimized = self._optimize_deterministic(question, task_type, model_name, template)
        
        return optimized
    
    def _classify_task(self, question: str) -> TaskType:
        """Classify question into task type using pattern matching."""
        question_lower = question.lower()
        
        scores = {task_type: 0 for task_type in TaskType}
        
        for task_type, patterns in self._task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    scores[task_type] += 1
        
        # Get the task type with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return TaskType.UNKNOWN
        
        for task_type, score in scores.items():
            if score == max_score:
                return task_type
        
        return TaskType.REASONING  # Default
    
    def _optimize_deterministic(
        self, 
        question: str, 
        task_type: TaskType, 
        model_name: str,
        template: Dict,
    ) -> OptimizedPrompt:
        """Optimize prompt using deterministic rules."""
        constraints = template["constraints"]
        output_format = template["output_format"]
        
        # Build the optimized prompt
        prompt_parts = []
        
        # Task-specific instructions
        if task_type == TaskType.MATH:
            # Detect special cases
            is_discount = any(kw in question.lower() for kw in ["discount", "off", "sale"])
            is_percent = any(kw in question.lower() for kw in ["percent", "%"])
            
            prompt_parts.append("MATHEMATICAL CALCULATION")
            prompt_parts.append("")
            
            if is_discount and is_percent:
                # Special handling for discount problems
                prompt_parts.append("DISCOUNT PROBLEM: Calculate FINAL PRICE after discount.")
                prompt_parts.append("Formula: FINAL = ORIGINAL × (1 - DISCOUNT/100)")
                prompt_parts.append("")
                prompt_parts.append(f"Question: {question}")
                prompt_parts.append("")
                prompt_parts.append("Example: $45 shirt, 30% off → 45 × 0.70 = 31.5")
                prompt_parts.append("")
                prompt_parts.append("CRITICAL: Give ONLY the number. No words, no explanation.")
                prompt_parts.append("")
                prompt_parts.append("FINAL PRICE = ")
            else:
                prompt_parts.append("Calculate the following. Give ONLY the numerical answer.")
                prompt_parts.append("")
                prompt_parts.append(f"Question: {question}")
                prompt_parts.append("")
                prompt_parts.append("Rules:")
                for c in constraints:
                    prompt_parts.append(f"- {c}")
                prompt_parts.append("")
                prompt_parts.append("CRITICAL: Give ONLY the number. No words, no explanation.")
                prompt_parts.append("")
                prompt_parts.append("ANSWER = ")
        
        elif task_type == TaskType.CODE:
            prompt_parts.append("CODE OUTPUT PREDICTION")
            prompt_parts.append("")
            prompt_parts.append("What does this code output? Give ONLY the output.")
            prompt_parts.append("")
            prompt_parts.append(f"Code: {question}")
            prompt_parts.append("")
            prompt_parts.append("Rules:")
            for c in constraints:
                prompt_parts.append(f"- {c}")
            prompt_parts.append("")
            prompt_parts.append("OUTPUT:")
        
        elif task_type == TaskType.REASONING:
            prompt_parts.append("LOGICAL REASONING")
            prompt_parts.append("")
            prompt_parts.append("Analyze this problem step by step.")
            prompt_parts.append("")
            prompt_parts.append(f"Problem: {question}")
            prompt_parts.append("")
            prompt_parts.append("Rules:")
            for c in constraints:
                prompt_parts.append(f"- {c}")
            prompt_parts.append("")
            prompt_parts.append("ANSWER:")
        
        else:
            prompt_parts.append(f"Question: {question}")
            prompt_parts.append("")
            prompt_parts.append("ANSWER:")
        
        optimized_prompt = "\n".join(prompt_parts)
        
        return OptimizedPrompt(
            original_question=question,
            task_type=task_type,
            optimized_prompt=optimized_prompt,
            constraints=constraints,
            output_format=output_format,
            confidence=0.8,
            metadata={"model": model_name, "method": "deterministic"},
        )
    
    def _optimize_with_llm(
        self, 
        question: str, 
        task_type: TaskType,
        model_name: str,
    ) -> OptimizedPrompt:
        """Optimize prompt using a small LLM (meta-prompting)."""
        # This would use a small LLM to generate better prompts
        # For now, fall back to deterministic
        return self._optimize_deterministic(question, task_type, model_name, 
                                            self._constraint_templates[task_type])


class DynamicConstraints:
    """Dynamic constraints that adapt based on context."""
    
    @staticmethod
    def for_math(question: str) -> List[str]:
        """Generate constraints for math problems."""
        constraints = [
            "GIVE ONLY THE NUMERICAL ANSWER",
            "NO explanation, NO steps",
        ]
        
        # Detect if decimal expected
        if any(kw in question.lower() for kw in ["percent", "price", "cost", "decimal"]):
            constraints.append("Use decimal point: 3.14")
        else:
            constraints.append("If integer, no decimal")
        
        # Detect if large number
        if re.search(r"\d{4,}", question):
            constraints.append("Use commas for thousands: 1,234")
        
        return constraints
    
    @staticmethod
    def for_code(question: str) -> List[str]:
        """Generate constraints for code problems."""
        constraints = [
            "GIVE ONLY THE OUTPUT",
            "NO explanation, NO code",
        ]
        
        # Detect Python specific
        if "python" in question.lower() or "print" in question.lower():
            constraints.append("Python output format")
        
        # Detect error possibility
        if any(kw in question for kw in ["error", "exception", "invalid"]):
            constraints.append("If error: ERROR: [reason]")
        
        return constraints
    
    @staticmethod
    def for_reasoning(question: str) -> List[str]:
        """Generate constraints for reasoning problems."""
        return [
            "Think step by step (max 3 steps)",
            "Be concise",
            "End with ANSWER: [short answer]",
            "Answer must be 1-5 words",
        ]


class PromptEnhancer:
    """Enhances prompts with additional context and constraints."""
    
    def __init__(self):
        self._context_db = {
            "sheep": "This is a classic riddle. Focus on what 'all but' means.",
            "machines": "This is about rate of production. Each machine works independently.",
            "widgets": "Consider the production rate per machine.",
            "discount": "Discount means SUBTRACT from original price. Use: price × (1 - discount/100)",
            "off": "Percent OFF means the price is REDUCED. Multiply original by (1 - percent/100)",
            "sale": "Sale price = original × (1 - discount percentage/100)",
        }
    
    def enhance(self, question: str, task_type: TaskType) -> str:
        """Enhance prompt with relevant context."""
        enhancement = []
        
        # Add context if available
        for keyword, context in self._context_db.items():
            if keyword in question.lower():
                enhancement.append(f"Hint: {context}")
                break
        
        # Add task-type specific enhancement
        if task_type == TaskType.MATH:
            enhancement.append("Perform the calculation directly.")
        elif task_type == TaskType.CODE:
            enhancement.append("Execute the code mentally and give the output.")
        elif task_type == TaskType.REASONING:
            enhancement.append("Analyze the logic carefully.")
        
        return "\n".join(enhancement) if enhancement else ""


# Example usage
if __name__ == "__main__":
    optimizer = PromptOptimizer(use_llm=False)
    
    test_questions = [
        "What is 127 + 389?",
        "What does this Python code output: print(2 ** 3 ** 2)?",
        "If all cats are animals, and some animals are black, can we conclude some cats are black?",
        "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
    ]
    
    for q in test_questions:
        result = optimizer.optimize(q)
        print(f"\n{'='*60}")
        print(f"Original: {q}")
        print(f"Task Type: {result.task_type.value}")
        print(f"Constraints: {result.constraints}")
        print(f"\nOptimized Prompt:")
        print(result.optimized_prompt)
