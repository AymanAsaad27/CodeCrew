# CodeCrew

**A local multi-agent AI code-fixing assistant built with Python, LangGraph, and Ollama.**

CodeCrew takes a coding task, creates a plan, executes steps using development tools, and checks the result. It is designed as a reproducible portfolio project demonstrating agent workflows, tool use, bounded execution, and automated testing with a locally hosted language model.

## Features

- **Planning and execution:** Separates task planning from tool execution.
- **Local LLM integration:** Uses Ollama to run a coding model locally.
- **Graph-based workflow:** Uses LangGraph to coordinate execution and control flow.
- **Development tools:** Supports repository inspection, file reading, code search, targeted file replacement, and test execution.
- **Bounded execution:** Limits agent actions to avoid uncontrolled execution loops.
- **Test verification:** Uses test results to determine whether a task can be marked completed.
- **Automated tests:** Includes tests for executor and graph behavior.

## Architecture

```text
User Task
   |
   v
Planner
   |
   v
LangGraph Workflow
   |
   v
Executor
   |
   +----> Repository and file tools
   |
   +----> Test runner
   |
   v
Record Results and Check Limits
   |
   v
Finalize Task Status
   |
   v
Final Answer
```

The planner determines the steps needed for a task. The executor carries out those steps through the available tools. The graph manages control flow and execution limits, while test results help determine whether the task was completed successfully.

## Technology Stack

- Python 3.12
- LangGraph
- LangChain Core
- Ollama
- Pydantic
- pytest

## Project Structure

```text
codecrew/
├── bench/
│   └── __init__.py
├── src/
│   └── codecrew/
│       ├── __init__.py
│       ├── executor.py
│       ├── graph.py
│       ├── llm.py
│       ├── main.py
│       ├── planner.py
│       ├── schemas.py
│       ├── state.py
│       └── tools.py
├── tests/
│   ├── test_executor.py
│   └── test_graph.py
├── .gitignore
└── requirements.txt
```

## Prerequisites

- Python 3.12
- Git
- [Ollama](https://ollama.com/)
- A compatible local language model

The examples below use `qwen2.5-coder:7b`, which was used during development.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AymanAsaad27/CodeCrew.git
cd CodeCrew
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and start the local model

Install Ollama from [ollama.com](https://ollama.com/), then download the model:

```bash
ollama pull qwen2.5-coder:7b
```

Make sure Ollama is running before starting CodeCrew.

## Usage

Run CodeCrew against a repository with a natural-language coding task:

```bash
python -m src.codecrew.main \
  --repo /path/to/your/repository \
  --task "Describe the bug and explain the expected behavior." \
  --max-actions 12
```

Replace `/path/to/your/repository` with the path to the repository you want CodeCrew to inspect or modify.

**Important:** Run the agent against a disposable copy or a Git branch when testing code changes. Review its modifications before using them in important projects.

## Example: Fixing a Bug

A demonstration task used during development:

```bash
python -m src.codecrew.main \
  --repo /tmp/codecrew-demo \
  --task "Fix the bug in calculator.py: the add(a, b) function subtracts instead of adding. Make the smallest necessary change and verify the fix by running the tests." \
  --max-actions 12
```

The demonstration repository contained a faulty implementation:

```python
def add(a, b):
    return a - b
```

CodeCrew replaced the subtraction with addition and ran the tests.

The verified demonstration completed with **2 passing tests** and a final status of `completed`.

## Running Tests

From the project root, with the virtual environment activated:

```bash
python -m pytest -q
```

The development test suite passed **11 tests** during the initial verification.

Run the tests again after making changes to confirm the current state.

## Design Goals

CodeCrew explores practical patterns for building AI agents that interact with software repositories:

- Explicit planning and execution stages
- Tool-based code modification
- Controlled execution and state management
- Automated verification of changes
- Local model inference
- Testable graph-based orchestration

## Limitations and Future Improvements

Potential future improvements include:

- More robust patch generation and validation
- Better recovery from tool and model failures
- Additional benchmark tasks and success metrics
- Structured execution logs and evaluation reports
- CI workflows for automated testing
- Improved safeguards for potentially destructive changes

These are proposed improvements, not claims that they are already implemented.

## Author

**Ayman Asaad**

GitHub: [@AymanAsaad27](https://github.com/AymanAsaad27)

## License

No license has been specified yet. If you intend to make this project open source, consider adding a license file.
