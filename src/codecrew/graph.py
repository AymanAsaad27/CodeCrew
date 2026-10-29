
from langgraph.graph import END, START, StateGraph

from src.codecrew.executor import executor_node
from src.codecrew.llm import LLM
from src.codecrew.planner import planner_node
from src.codecrew.state import CodeCrewState


MAX_EXECUTOR_BATCHES = 4


def build_graph(llm: LLM):
    """Build the bounded CodeCrew planner-executor workflow."""

    workflow = StateGraph(CodeCrewState)

    workflow.add_node(
        "planner",
        lambda state: planner_node(state, llm),
    )
    workflow.add_node(
        "executor",
        lambda state: executor_node(state, llm),
    )

    workflow.add_edge(START, "planner")

    def route_after_planning(state: CodeCrewState) -> str:
        if state.get("status") == "planned":
            return "executor"
        return "end"

    workflow.add_conditional_edges(
        "planner",
        route_after_planning,
        {
            "executor": "executor",
            "end": END,
        },
    )

    def route_after_execution(state: CodeCrewState) -> str:
        if state.get("status") != "executing":
            return "end"

        if state.get("executor_batches", 0) >= MAX_EXECUTOR_BATCHES:
            return "limit"

        return "executor"

    def record_executor_batch(state: CodeCrewState) -> dict:
        return {
            "executor_batches": state.get("executor_batches", 0) + 1,
        }

    def handle_batch_limit(state: CodeCrewState) -> dict:
        return {
            "status": "failed",
            "error": (
                f"Executor batch limit reached ({MAX_EXECUTOR_BATCHES}). "
                "The task may be incomplete."
            ),
            "final_answer": (
                "Stopped because the executor reached its batch limit. "
                "Review the recorded actions and tool results."
            ),
        }

    def finalize_result(state: CodeCrewState) -> dict:
        """Finalize the workflow without overstating success."""

        if state.get("status") != "investigated":
            return {}

        tool_results = state.get("tool_results", [])

        tests_passed = any(
            item.get("tool") == "run_tests"
            and isinstance(item.get("result"), dict)
            and item["result"].get("passed") is True
            for item in tool_results
        )

        if tests_passed:
            return {
                "status": "completed",
                "final_answer": (
                    "The task finished successfully, and the tests passed."
                ),
            }

        return {
            "status": "investigated",
            "final_answer": (
                "The investigation finished, but passing tests were not "
                "verified. Review the recorded actions and tool results."
            ),
        }

    workflow.add_node("record_batch", record_executor_batch)
    workflow.add_node("batch_limit", handle_batch_limit)
    workflow.add_node("finalize", finalize_result)

    workflow.add_edge("executor", "record_batch")

    workflow.add_conditional_edges(
        "record_batch",
        route_after_execution,
        {
            "executor": "executor",
            "end": "finalize",
            "limit": "batch_limit",
        },
    )

    workflow.add_edge("batch_limit", END)
    workflow.add_edge("finalize", END)

    return workflow.compile()
