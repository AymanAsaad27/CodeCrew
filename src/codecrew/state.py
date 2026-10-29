from typing import Any, TypedDict


class CodeCrewState(TypedDict, total=False):
    """Shared state passed between CodeCrew workflow nodes."""

    task: str
    repo_path: str

    plan: dict[str, Any]
    current_step: int

    messages: list[dict[str, str]]
    actions: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    test_results: dict[str, Any]
    review: dict[str, Any]
    executor_batches: int
    iteration: int
    max_iterations: int

    status: str
    final_answer: str

    error: str
