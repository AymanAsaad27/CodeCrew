
import json
from pathlib import Path
from typing import Any

from src.codecrew.llm import BudgetExceeded, LLM
from src.codecrew.schemas import Action
from src.codecrew.state import CodeCrewState
from src.codecrew.tools import (
    list_dir,
    read_file,
    replace_in_file,
    run_tests,
    search,
)


MAX_ACTIONS = 12



def _tool_result(
    root: str,
    action: Action,
) -> dict[str, Any]:
    """Execute one validated action and return a serializable result."""

    # Directory paths should be relative to the repository root.
    if action.tool == "list_dir":
        relative_path = action.path or "."

        if Path(relative_path).is_absolute():
            try:
                relative_path = str(
                    Path(relative_path).resolve().relative_to(
                        Path(root).resolve()
                    )
                )
            except ValueError:
                raise ValueError(
                    "Directory path must stay inside the repository root"
                )

        result = list_dir(root, relative_path)

    elif action.tool == "read_file":
        result = read_file(root, action.path)

    elif action.tool == "search":
        result = search(root, action.query)

    elif action.tool == "replace_in_file":
        result = replace_in_file(
            root,
            action.path,
            action.old,
            action.new,
        )

    elif action.tool == "run_tests":
        result = run_tests(root)

    elif action.tool == "finish":
        result = {"finished": True}

    else:
        raise ValueError(f"Unsupported tool: {action.tool}")

    return {
        "tool": action.tool,
        "thought": action.thought,
        "path": action.path,
        "result": result,
    }



def executor_node(
    state: CodeCrewState,
    llm: LLM,
) -> dict[str, Any]:
    """Perform bounded, evidence-grounded repository actions."""

    root = state.get("repo_path", "").strip()
    task = state.get("task", "").strip()

    if not root:
        return {
            "status": "failed",
            "error": "Repository path cannot be empty.",
        }

    repo = Path(root).resolve()

    if not repo.is_dir():
        return {
            "status": "failed",
            "error": f"Repository directory does not exist: {repo}",
        }

    if not task:
        return {
            "status": "failed",
            "error": "Task description cannot be empty.",
        }

    actions = list(state.get("actions", []))
    results = list(state.get("tool_results", []))
    max_actions = state.get("max_iterations", MAX_ACTIONS)

    if not isinstance(max_actions, int) or isinstance(max_actions, bool):
        return {
            "status": "failed",
            "error": "max_iterations must be an integer.",
            "actions": actions,
            "tool_results": results,
        }

    if max_actions < 1:
        return {
            "status": "failed",
            "error": "max_iterations must be at least 1.",
            "actions": actions,
            "tool_results": results,
        }

    if len(actions) >= max_actions:
        return {
            "status": "failed",
            "error": f"Action limit reached ({max_actions}).",
            "actions": actions,
            "tool_results": results,
            "iteration": len(actions),
        }

    plan = state.get("plan", {})
    executed_this_call = 0

    while len(actions) < max_actions:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are CodeCrew's repository investigation executor. "
                    "Choose exactly one next tool action. Never claim to have "
                    "inspected files unless tool results show that. Prefer "
                    "listing and reading files before editing. Only edit "
                    "using exact source text supported by prior read results. "
                    "Do not edit tests just to make them pass. Use run_tests "
                    "to verify changes. Use finish when the investigation or "
                    "repair is complete, or when it cannot proceed safely. "
                    "Return the required structured schema."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "repository": str(repo),
                        "plan": plan,
                        "previous_actions": actions,
                        "tool_results": results,
                        "allowed_tools": [
                            "list_dir",
                            "read_file",
                            "search",
                            "replace_in_file",
                            "run_tests",
                            "finish",
                        ],
                        "instruction": (
    "Choose one useful next action. Start by listing "
    "the repository root if its contents are not "
    "already recorded. Inspect source before editing. "
    "For replace_in_file, ALWAYS set path to the target "
    "file, old to the exact non-empty text to replace, "
    "and new to the replacement text. Do not put the "
    "old text in query. The old text must match the file "
    "exactly and should identify only one occurrence. "
    "After editing, run the tests and inspect the result. "
    "If a tool returns an error, correct the arguments "
    "before retrying."
),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        try:
            action = llm.json(
                model="qwen2.5-coder:7b",
                messages=messages,
                schema=Action,
                temperature=0.0,
            )

            record = _tool_result(str(repo), action)
            actions.append(action.model_dump())
            results.append(record)
            executed_this_call += 1

        
        except (BudgetExceeded, ValueError, OSError) as exc:
            error_message = f"{type(exc).__name__}: {exc}"

            # Preserve the attempted action when the tool execution failed.
            if "action" in locals() and isinstance(action, Action):
                actions.append(action.model_dump())
                results.append({
                    "tool": action.tool,
                    "thought": action.thought,
                    "path": action.path,
                    "result": {"error": error_message},
                })

            return {
                "status": "failed",
                "error": f"Executor failed: {error_message}",
                "actions": actions,
                "tool_results": results,
                "iteration": len(actions),
            }


        except Exception as exc:
            return {
                "status": "failed",
                "error": (
                    f"Unexpected executor error: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "actions": actions,
                "tool_results": results,
                "iteration": len(actions),
            }

        if action.tool == "finish":
            return {
                "actions": actions,
                "tool_results": results,
                "iteration": len(actions),
                "status": "investigated",
                "error": "",
            }

        # Stop after a bounded batch so a workflow can regain control.
        if executed_this_call >= 3:
            return {
                "actions": actions,
                "tool_results": results,
                "iteration": len(actions),
                "status": "executing",
                "error": "",
            }

    return {
        "actions": actions,
        "tool_results": results,
        "iteration": len(actions),
        "status": "failed",
        "error": f"Action limit reached ({max_actions}).",
    }
