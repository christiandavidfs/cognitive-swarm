"""Simple test to verify the system works."""

from cognitive_swarm.models.backend import DummyBackend, ModelConfig
from cognitive_swarm.agents.solver import SolverAgent
from cognitive_swarm.agents.critic import CriticAgent
from cognitive_swarm.agents.base import AgentContext


def test_basic_workflow():
    """Test basic agent workflow with dummy backend."""
    # Create dummy backend
    backend = DummyBackend(responses=[
        "The answer is 42. This is because 6*7=42.",
        "I disagree. The calculation shows 6*7=42, but let me verify.",
        "After verification, I confirm 6*7=42.",
    ])
    backend.load_model(ModelConfig(model_path="dummy"))

    # Create agents
    solver_a = SolverAgent("solver_a", backend, strategy="normal")
    solver_b = SolverAgent("solver_b", backend, strategy="skeptical")
    critic = CriticAgent("critic", backend)

    # Create context
    context = AgentContext(task="What is 6 * 7?")

    # Run solvers
    response_a = solver_a.run(context)
    response_b = solver_b.run(context)

    print(f"Solver A: {response_a.content[:50]}...")
    print(f"Solver B: {response_b.content[:50]}...")

    # Update context with solutions
    context.history.append({
        "agent": "solver_a",
        "content": response_a.content,
    })
    context.history.append({
        "agent": "solver_b",
        "content": response_b.content,
    })

    # Run critic
    critique = critic.run(context)
    print(f"Critic: {critique.content[:50]}...")

    print("\nBasic workflow test passed!")


def test_evaluation():
    """Test evaluation framework."""
    from cognitive_swarm.evaluation import BenchmarkSuite

    suite = BenchmarkSuite()
    problems = suite.get_problems()

    print(f"\nBenchmark suite loaded: {len(problems)} problems")
    for p in problems[:3]:
        print(f"  {p.id}: {p.question[:30]}...")

    print("Evaluation test passed!")


if __name__ == "__main__":
    test_basic_workflow()
    test_evaluation()
