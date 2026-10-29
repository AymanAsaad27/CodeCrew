from src.codecrew.llm import LLM
from src.codecrew.schemas import Plan
from src.codecrew.state import CodeCrewState


def planner_node(state: CodeCrewState, llm: LLM) -> dict:
    """Generate a structured debugging plan for a reported issue."""

    task = state.get("task", "").strip()
    repo_path = state.get("repo_path", "").strip()

    if not task:
        return {
            "status": "failed",
            "error": "The task description cannot be empty.",
        }

    if not repo_path:
        return {
            "status": "failed",
            "error": "The repository path cannot be empty.",
        }

    messages = [
        {
            "role": "system",
            "content": (
                "You are the planning component of CodeCrew, a code-fixing "
                "assistant. Produce a concise, evidence-conscious debugging "
                "plan. Do not claim to have inspected files. File names and "
                "steps are hypotheses until verified by repository tools."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Repository path: {repo_path}\n"
                f"Reported bug: {task}\n\n"
                "Create a plan for investigating and fixing this bug. "
                "List likely files to inspect, but label your reasoning "
                "as provisional. Do not modify files."
            ),
        },
    ]

    try:
        plan = llm.json(
            model="qwen2.5-coder:7b",
            messages=messages,
            schema=Plan,
            temperature=0.0,
        )

        return {
            "plan": plan.model_dump(),
            "current_step": 0,
            "status": "planned",
            "error": "",
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error": f"Planner failed: {type(exc).__name__}: {exc}",
        }
