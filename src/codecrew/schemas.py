from typing import Literal

from pydantic import BaseModel, Field


class Action(BaseModel):
    """One tool action requested by an agent."""

    thought: str = Field(description="Brief reason for the action")
    tool: Literal[
        "list_dir",
        "read_file",
        "search",
        "replace_in_file",
        "run_tests",
        "finish",
    ]
    path: str = ""
    query: str = ""
    old: str = ""
    new: str = ""


class Plan(BaseModel):
    """A concise plan for investigating and fixing a reported bug."""

    hypothesis: str
    files_to_inspect: list[str]
    steps: list[str]


class Verdict(BaseModel):
    """Code review decision."""

    approve: bool
    feedback: str
