"""Main entry point for the cognitive swarm."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive_swarm.orchestration.workflow import SwarmOrchestrator, SwarmConfig
from cognitive_swarm.evaluation import BenchmarkSuite, Evaluator


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('cognitive_swarm.log'),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Cognitive Swarm - Local Multi-Agent Reasoning")
    parser.add_argument("--model", type=str, help="Path to the model")
    parser.add_argument("--task", type=str, help="Task to solve")
    parser.add_argument("--mode", choices=["single", "debate", "verified", "full"],
                       default="debate", help="Execution mode")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--max-rounds", type=int, default=2, help="Max debate rounds")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per response")

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Default model path (will be overridden by args)
    model_path = args.model or "mlx-community/phi-3.5-mini-instruct-4bit"

    # Create config
    config = SwarmConfig(
        model_path=model_path,
        max_rounds=args.max_rounds,
        max_tokens=args.max_tokens,
    )

    # Create orchestrator
    logger.info(f"Initializing swarm with model: {model_path}")
    orchestrator = SwarmOrchestrator(config)

    if args.evaluate:
        # Run evaluation suite
        logger.info("Running evaluation suite")
        benchmark = BenchmarkSuite()
        evaluator = Evaluator(orchestrator, benchmark)

        modes = ["single", "debate", "verified"]
        table = evaluator.evaluate_all(modes, max_problems=5)
        evaluator.print_comparison(table)

    elif args.task:
        # Run single task
        logger.info(f"Running task: {args.task}")
        result = orchestrator.run(args.task, mode=args.mode)

        print("\n" + "=" * 60)
        print("TASK:", args.task)
        print("MODE:", args.mode)
        print("=" * 60)
        print("\nFINAL ANSWER:")
        print(result.final_answer)
        print(f"\nCONFIDENCE: {result.confidence:.2f}")
        print(f"ROUNDS: {result.debate_result.total_rounds}")
        print(f"LATENCY: {result.metrics.get('latency_ms', 0):.0f}ms")
        print("=" * 60)

    else:
        # Interactive mode
        print("Cognitive Swarm - Interactive Mode")
        print("Type 'quit' to exit, 'evaluate' to run benchmarks")
        print()

        while True:
            try:
                task = input("Enter task: ").strip()
                if task.lower() == 'quit':
                    break
                if task.lower() == 'evaluate':
                    benchmark = BenchmarkSuite()
                    evaluator = Evaluator(orchestrator, benchmark)
                    modes = ["single", "debate", "verified"]
                    table = evaluator.evaluate_all(modes, max_problems=3)
                    evaluator.print_comparison(table)
                    continue
                if not task:
                    continue

                result = orchestrator.run(task, mode=args.mode)

                print("\n" + "-" * 40)
                print("ANSWER:", result.final_answer)
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Rounds: {result.debate_result.total_rounds}")
                print("-" * 40)
                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    # Cleanup
    orchestrator.unload()


if __name__ == "__main__":
    main()
