
from pathlib import Path

import pytest

from src.codecrew.executor import _tool_result
from src.codecrew.schemas import Action


def test_list_dir_returns_real_repository_files(tmp_path: Path):
    (tmp_path / "cart.py").write_text(
        "def calculate_total(prices):\n    return sum(prices) + 1\n",
        encoding="utf-8",
    )
    (tmp_path / "test_cart.py").write_text(
        "def test_example():\n    assert True\n",
        encoding="utf-8",
    )

    action = Action(
        thought="Inspect the repository before investigating the bug.",
        tool="list_dir",
        path=".",
    )

    result = _tool_result(str(tmp_path), action)

    assert result["tool"] == "list_dir"
    assert "FILE cart.py" in result["result"]
    assert "FILE test_cart.py" in result["result"]


def test_list_dir_normalizes_absolute_path_inside_repository(
    tmp_path: Path,
):
    (tmp_path / "cart.py").write_text("pass\n", encoding="utf-8")

    action = Action(
        thought="Inspect the repository.",
        tool="list_dir",
        path=str(tmp_path.resolve()),
    )

    result = _tool_result(str(tmp_path), action)

    assert "FILE cart.py" in result["result"]


def test_list_dir_rejects_absolute_path_outside_repository(
    tmp_path: Path,
):
    outside = tmp_path.parent / "outside-codecrew-test-dir"

    action = Action(
        thought="Try an outside directory.",
        tool="list_dir",
        path=str(outside.resolve()),
    )

    with pytest.raises(ValueError, match="inside the repository root"):
        _tool_result(str(tmp_path), action)


def test_read_file_returns_actual_source(tmp_path: Path):
    source = "def calculate_total(prices):\n    return sum(prices) + 1\n"
    (tmp_path / "cart.py").write_text(source, encoding="utf-8")

    action = Action(
        thought="Inspect the faulty function.",
        tool="read_file",
        path="cart.py",
    )

    result = _tool_result(str(tmp_path), action)

    assert result["result"] == source


def test_finish_records_completion_without_editing_files(tmp_path: Path):
    source_path = tmp_path / "cart.py"
    source_path.write_text("original\n", encoding="utf-8")

    action = Action(
        thought="Investigation is complete.",
        tool="finish",
    )

    result = _tool_result(str(tmp_path), action)

    assert result["result"] == {"finished": True}
    assert source_path.read_text(encoding="utf-8") == "original\n"


from src.codecrew.executor import executor_node


class FakeLLM:
    """Return predefined actions without calling a real model."""

    def __init__(self, actions):
        self.actions = iter(actions)

    def json(self, **kwargs):
        return next(self.actions)


def test_executor_performs_multiple_actions_and_finishes(tmp_path):
    (tmp_path / "example.py").write_text(
        "def greet():\n    return 'hello'\n",
        encoding="utf-8",
    )

    fake_llm = FakeLLM([
        Action(
            thought="Inspect repository files.",
            tool="list_dir",
            path=".",
        ),
        Action(
            thought="Read the source file.",
            tool="read_file",
            path="example.py",
        ),
        Action(
            thought="Investigation is complete.",
            tool="finish",
        ),
    ])

    state = {
        "task": "Inspect the example greeting function.",
        "repo_path": str(tmp_path),
        "plan": {"steps": ["Inspect files", "Read source"]},
        "actions": [],
        "tool_results": [],
        "max_iterations": 5,
    }

    result = executor_node(state, fake_llm)

    assert result["status"] == "investigated"
    assert [action["tool"] for action in result["actions"]] == [
        "list_dir",
        "read_file",
        "finish",
    ]
    assert len(result["tool_results"]) == 3
    assert "def greet()" in result["tool_results"][1]["result"]


def test_executor_respects_action_limit(tmp_path):
    fake_llm = FakeLLM([
        Action(
            thought="Inspect repository.",
            tool="list_dir",
            path=".",
        ),
    ])

    state = {
        "task": "Inspect the repository.",
        "repo_path": str(tmp_path),
        "plan": {},
        "actions": [],
        "tool_results": [],
        "max_iterations": 1,
    }

    result = executor_node(state, fake_llm)

    assert result["status"] == "failed"
    assert len(result["actions"]) == 1
    assert "Action limit reached" in result["error"]


def test_executor_reports_invalid_path_without_leaving_repository(tmp_path):
    fake_llm = FakeLLM([
        Action(
            thought="Attempt to inspect a file outside the repository.",
            tool="read_file",
            path="../outside.txt",
        ),
    ])

    state = {
        "task": "Inspect the repository.",
        "repo_path": str(tmp_path),
        "plan": {},
        "actions": [],
        "tool_results": [],
        "max_iterations": 3,
    }

    result = executor_node(state, fake_llm)

    
    assert result["status"] == "failed"
    assert "Executor failed" in result["error"]

    # Preserve the attempted action for debugging.
    assert len(result["actions"]) == 1
    assert result["actions"][0]["tool"] == "read_file"
    assert result["actions"][0]["path"] == "../outside.txt"

    # Record the failure, not a successful file read.
    assert len(result["tool_results"]) == 1
    assert "error" in result["tool_results"][0]["result"]

