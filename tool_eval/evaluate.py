"""CLI entry point for tool-use evaluation.

Usage:
    python -m tool_eval.evaluate \
        --benchmark tau2-bench \
        --backend openai \
        --model gpt-4o \
        --limit 10

    python -m tool_eval.evaluate \
        --benchmark swe-bench \
        --backend anthropic \
        --model claude-sonnet-4-20250514 \
        --limit 5

    python -m tool_eval.evaluate \
        --benchmark livecodebench \
        --backend vllm \
        --model Qwen/Qwen2.5-72B-Instruct \
        --base-url http://localhost:8000/v1
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Tool-use benchmark evaluation")
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark name: tau2-bench, swe-bench, livecodebench, terminalbench, browsecomp, mcp-atlas")
    parser.add_argument("--backend", required=True,
                        help="Model backend: openai, anthropic, vllm")
    parser.add_argument("--model", required=True,
                        help="Model name/path")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: from env var)")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (for vLLM/custom endpoints)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of tasks to evaluate")
    parser.add_argument("--task-ids", nargs="+", default=None,
                        help="Specific task IDs to evaluate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens per generation")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Max agent turns per task")
    parser.add_argument("--output-dir", default="./tool_eval_results",
                        help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    # Benchmark-specific args
    parser.add_argument("--domain", default="retail",
                        help="tau2-bench domain: airline, retail, telecom")
    parser.add_argument("--search-backend", default="serper",
                        help="BrowseComp search backend: serper")

    # W&B integration
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity/team")

    args = parser.parse_args()

    # Import here to register environments
    import tool_eval.environments.tau2_bench  # noqa: F401
    import tool_eval.environments.swe_bench  # noqa: F401
    import tool_eval.environments.livecodebench  # noqa: F401
    import tool_eval.environments.terminalbench  # noqa: F401
    import tool_eval.environments.browsecomp  # noqa: F401
    import tool_eval.environments.mcp_atlas  # noqa: F401

    from tool_eval.agent import AgentLoop
    from tool_eval.environments import get_environment
    from tool_eval.models import create_backend

    # Create model backend
    logger.info(f"Creating {args.backend} backend with model {args.model}")
    model = create_backend(
        backend_type=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Create environment
    env_kwargs = {}
    if args.benchmark == "tau2-bench":
        env_kwargs["domain"] = args.domain
    elif args.benchmark == "browsecomp":
        env_kwargs["search_backend"] = args.search_backend

    logger.info(f"Creating environment: {args.benchmark}")
    env = get_environment(args.benchmark, **env_kwargs)

    # Create agent loop
    agent = AgentLoop(
        model=model,
        environment=env,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    # Run evaluation
    start = time.time()
    results = agent.run_all(
        task_ids=args.task_ids,
        limit=args.limit,
    )
    elapsed = time.time() - start

    # Compute aggregate metrics
    total = len(results)
    successful = sum(1 for r in results.values() if r.success)
    avg_score = sum(r.score for r in results.values()) / total if total else 0
    avg_turns = sum(r.num_turns for r in results.values()) / total if total else 0
    avg_tools = sum(r.num_tool_calls for r in results.values()) / total if total else 0
    total_tokens = sum(r.metadata.get("total_tokens", 0) for r in results.values())

    summary = {
        "benchmark": args.benchmark,
        "model": args.model,
        "backend": args.backend,
        "num_tasks": total,
        "num_successful": successful,
        "success_rate": successful / total if total else 0,
        "avg_score": avg_score,
        "avg_turns": avg_turns,
        "avg_tool_calls": avg_tools,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
    }

    logger.info("=" * 60)
    logger.info(f"Benchmark:    {args.benchmark}")
    logger.info(f"Model:        {args.model}")
    logger.info(f"Tasks:        {total}")
    logger.info(f"Success rate: {successful}/{total} ({summary['success_rate']:.1%})")
    logger.info(f"Avg score:    {avg_score:.3f}")
    logger.info(f"Avg turns:    {avg_turns:.1f}")
    logger.info(f"Avg tools:    {avg_tools:.1f}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Elapsed:      {elapsed:.1f}s")
    logger.info("=" * 60)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_slug = args.model.replace("/", "--")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.benchmark}_{model_slug}_{timestamp}"

    # Summary
    summary_path = output_dir / f"{prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Per-task results
    details = {}
    for task_id, result in results.items():
        details[task_id] = {
            "task_id": result.task_id,
            "success": result.success,
            "score": result.score,
            "num_turns": result.num_turns,
            "num_tool_calls": result.num_tool_calls,
            "metadata": result.metadata,
            "trajectory": result.trajectory,
        }

    details_path = output_dir / f"{prefix}_details.json"
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2)
    logger.info(f"Details saved to {details_path}")

    # W&B upload
    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{args.benchmark}/{model_slug}",
                config=vars(args),
            )
            wandb.log(summary)
            wandb.finish()
            logger.info(f"Results uploaded to W&B: {args.wandb_project}")
        except Exception as e:
            logger.warning(f"W&B upload failed: {e}")


if __name__ == "__main__":
    main()
