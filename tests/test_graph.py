
from src.codecrew.graph import build_graph
from src.codecrew.schemas import Action, Plan


class FakeLLM:
    """Return predefined structured responses without calling Ollama."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def json(self, **kwargs):
        self.calls += 1
        return next(self.responses)


def test_real_graph_runs_planner_and_executor(tmp_path):
    (tmp_path / "example.py").write_text(
        "def greet():\n    return 'hello'\n",
        encoding="utf-8",
    )

    llm = FakeLLM([
        Plan(
            hypothesis="Inspect the greeting function.",
            files_to_inspect=["example.py"],
            steps=["List files", "Read source"],
        ),
        Action(
            thought="Inspect repository contents.",
            tool="list_dir",
            path=".",
        ),
        Action(
            thought="Read the source file.",
            tool="read_file",
            path="example.py",
        ),
        Action(
            thought="The investigation is complete.",
            tool="finish",
        ),
    ])

    graph = build_graph(llm)
    result = graph.invoke({
        "task": "Inspect the greeting function.",
        "repo_path": str(tmp_path),
        "actions": [],
        "tool_results": [],
        "max_iterations": 5,
        "executor_batches": 0,
        "status": "starting",
    })

    assert result["status"] == "investigated"
    assert [action["tool"] for action in result["actions"]] == [
        "list_dir",
        "read_file",
        "finish",
    ]
    assert llm.calls == 4


def test_real_graph_stops_when_planning_fails():
    class FailingLLM:
        def json(self, **kwargs):
            raise RuntimeError("Simulated planning failure")

    graph = build_graph(FailingLLM())
    result = graph.invoke({
        "task": "Inspect a bug.",
        "repo_path": "/tmp",
        "actions": [],
        "tool_results": [],
        "status": "starting",
    })

    assert result["status"] == "failed"
    assert "Planner failed" in result["error"]
    assert result.get("actions", []) == []


def test_real_graph_stops_at_executor_batch_limit(tmp_path):
    from src.codecrew.graph import MAX_EXECUTOR_BATCHES

    (tmp_path / "example.py").write_text(
        "def greet():\n    return 'hello'\n",
        encoding="utf-8",
    )

    # Each batch asks for three actions, none of which finishes the task.
    responses = [
        Plan(
            hypothesis="Inspect the greeting function.",
            files_to_inspect=["example.py"],
            steps=["Inspect repository files"],
        )
    ]

    for _ in range(MAX_EXECUTOR_BATCHES * 3):
        responses.append(
            Action(
                thought="Continue investigating without finishing.",
                tool="list_dir",
                path=".",
            )
        )

    llm = FakeLLM(responses)
    graph = build_graph(llm)

    result = graph.invoke({
        "task": "Inspect the greeting function.",
        "repo_path": str(tmp_path),
        "actions": [],
        "tool_results": [],
        "max_iterations": 20,
        "executor_batches": 0,
        "status": "starting",
    })

    assert result["status"] == "failed"
    assert result["executor_batches"] == MAX_EXECUTOR_BATCHES
    assert "batch limit" in result["error"].lower()
    assert len(result["actions"]) == MAX_EXECUTOR_BATCHES * 3
