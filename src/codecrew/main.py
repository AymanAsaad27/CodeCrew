
import argparse
import json
import logging
from pathlib import Path

from src.codecrew.graph import build_graph
from src.codecrew.llm import LLM


def main():
    parser = argparse.ArgumentParser(
        description="CodeCrew: a local multi-agent code-fixing assistant."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the repository to inspect.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Description of the bug or coding task.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=12,
        help="Maximum number of executor actions.",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo).expanduser().resolve()

    if not repo_path.is_dir():
        parser.error(f"Repository directory does not exist: {repo_path}")

    if args.max_actions < 1:
        parser.error("--max-actions must be at least 1")

    logging.basicConfig(
        level=logging.ERROR,
        format="%(levelname)s: %(message)s",
    )

    try:
        llm = LLM(max_calls=30)
        graph = build_graph(llm)

        initial_state = {
            "task": args.task,
            "repo_path": str(repo_path),
            "actions": [],
            "tool_results": [],
            "max_iterations": args.max_actions,
            "iteration": 0,
            "executor_batches": 0,
            "status": "starting",
            "error": "",
        }

        result = graph.invoke(initial_state)

    except KeyboardInterrupt:
        print("\nCodeCrew interrupted.")
        raise SystemExit(130)

    except Exception as exc:
        logging.error(
            "CodeCrew failed unexpectedly: %s: %s",
            type(exc).__name__,
            exc,
        )
        raise SystemExit(1)

    print("\n=== CodeCrew Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
