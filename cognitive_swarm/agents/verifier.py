"""Verifier agent - uses deterministic tools to verify claims."""

import json
import re
import subprocess
import tempfile
import os
from typing import Any, Optional
from .base import BaseAgent, AgentContext, AgentResponse


class VerifierAgent(BaseAgent):
    """Verifier that uses deterministic methods to check claims.

    This agent does NOT use LLM for verification when possible.
    Instead, it uses:
    - Python execution for calculations
    - JSON validation
    - Regex matching
    - Unit tests
    - Code execution
    """

    def build_prompt(self, context: AgentContext, **kwargs) -> str:
        """Build verifier prompt - asks LLM to extract verifiable claims."""
        solutions = self._format_for_verification(context)

        return f"""You are a verification assistant. Your job is to extract VERIFIABLE claims from solutions.

Problem: {context.task}

Solutions to verify:
{solutions}

For each solution, extract:
1. Mathematical claims (e.g., "2+2=4", "x=5")
2. Factual claims that can be checked
3. Logical steps that can be validated
4. Any code or calculations mentioned

Format each claim as:
CLAIM: [the claim]
TYPE: [math/logic/fact/code]
CAN_VERIFY: [yes/no]
VERIFICATION_METHOD: [how to verify it]

Be specific and precise. Extract exact claims, not vague statements."""

    def parse_response(self, response: str, context: AgentContext) -> AgentResponse:
        """Parse verifier response and extract claims."""
        claims = self._extract_claims(response)

        # Try to verify mathematical claims
        verification_results = []
        for claim in claims:
            if claim.get("type") == "math" and claim.get("can_verify"):
                result = self._verify_math_claim(claim["claim"])
                verification_results.append(result)

        return AgentResponse(
            agent_name=self.name,
            content=response,
            confidence=0.9 if verification_results else 0.5,
            reasoning=response,
            metadata={
                "claims_extracted": len(claims),
                "claims_verifiable": sum(1 for c in claims if c.get("can_verify")),
                "verification_results": verification_results,
            },
        )

    def _extract_claims(self, text: str) -> list:
        """Extract claims from verifier response."""
        claims = []
        lines = text.split("\n")

        current_claim = {}
        for line in lines:
            line = line.strip()
            if line.startswith("CLAIM:"):
                if current_claim:
                    claims.append(current_claim)
                current_claim = {"claim": line[6:].strip()}
            elif line.startswith("TYPE:"):
                current_claim["type"] = line[5:].strip().lower()
            elif line.startswith("CAN_VERIFY:"):
                current_claim["can_verify"] = line[11:].strip().lower() == "yes"
            elif line.startswith("VERIFICATION_METHOD:"):
                current_claim["method"] = line[20:].strip()

        if current_claim:
            claims.append(current_claim)

        return claims

    def _verify_math_claim(self, claim: str) -> dict:
        """Try to verify a mathematical claim using Python."""
        try:
            # Clean up the claim
            claim_clean = claim.strip()

            # Try to evaluate as Python expression
            # Only allow safe operations
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max}

            # Check if it's a simple equation
            if "=" in claim_clean and not claim_clean.startswith("="):
                parts = claim_clean.split("=")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()

                    # Try to evaluate both sides
                    try:
                        left_val = eval(left, {"__builtins__": {}}, allowed_names)
                        right_val = eval(right, {"__builtins__": {}}, allowed_names)
                        is_correct = abs(left_val - right_val) < 1e-10
                        return {
                            "claim": claim,
                            "verified": True,
                            "correct": is_correct,
                            "left": left_val,
                            "right": right_val,
                        }
                    except:
                        pass

            # Try to evaluate as expression
            try:
                result = eval(claim_clean, {"__builtins__": {}}, allowed_names)
                return {
                    "claim": claim,
                    "verified": True,
                    "correct": bool(result),
                    "result": result,
                }
            except:
                pass

            # Try as code execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f"result = {claim_clean}\nprint(result)")
                f.flush()
                try:
                    output = subprocess.run(
                        ["python3", f.name],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if output.returncode == 0:
                        return {
                            "claim": claim,
                            "verified": True,
                            "correct": True,
                            "output": output.stdout.strip(),
                        }
                finally:
                    os.unlink(f.name)

            return {
                "claim": claim,
                "verified": False,
                "reason": "Could not parse or execute claim",
            }

        except Exception as e:
            return {
                "claim": claim,
                "verified": False,
                "error": str(e),
            }

    def _format_for_verification(self, context: AgentContext) -> str:
        """Format context for verification."""
        if not context.history:
            return "No solutions to verify."

        parts = []
        for entry in context.history:
            agent = entry.get("agent", "unknown")
            content = entry.get("content", "")
            parts.append(f"[{agent}]:\n{content}")

        return "\n\n".join(parts)

    def verify_json(self, text: str) -> dict:
        """Verify if text contains valid JSON."""
        try:
            # Try to extract JSON from text
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                json.loads(json_match.group())
                return {"valid": True, "json": json_match.group()}
            return {"valid": False, "reason": "No JSON found"}
        except json.JSONDecodeError as e:
            return {"valid": False, "reason": str(e)}

    def verify_code(self, code: str, expected_output: str = None) -> dict:
        """Verify code by executing it."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                try:
                    output = subprocess.run(
                        ["python3", f.name],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if expected_output:
                        return {
                            "success": output.returncode == 0,
                            "output": output.stdout.strip(),
                            "matches_expected": output.stdout.strip() == expected_output,
                        }
                    return {
                        "success": output.returncode == 0,
                        "output": output.stdout.strip(),
                        "error": output.stderr.strip() if output.returncode != 0 else None,
                    }
                finally:
                    os.unlink(f.name)
        except Exception as e:
            return {"success": False, "error": str(e)}
