from pathlib import Path


IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
}

MAX_FILE_CHARS = 20_000


def _safe_path(root: str | Path, relative_path: str = ".") -> Path:
    """Resolve a path and reject paths outside the repository root."""
    repo_root = Path(root).resolve()
    target = (repo_root / relative_path).resolve()

    if target != repo_root and repo_root not in target.parents:
        raise ValueError("Path must stay inside the repository root")

    return target


def list_dir(root: str | Path, path: str = ".") -> list[str]:
    """List files and directories relative to the repository root."""
    target = _safe_path(root, path)

    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    entries = []

    for item in sorted(target.iterdir(), key=lambda p: p.name.lower()):
        if item.name in IGNORED_DIRS:
            continue

        kind = "DIR " if item.is_dir() else "FILE"
        relative = item.relative_to(Path(root).resolve())
        entries.append(f"{kind} {relative}")

    return entries


def read_file(root: str | Path, path: str) -> str:
    """Read a UTF-8 text file, with a size limit."""
    target = _safe_path(root, path)

    if not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if target.stat().st_size > MAX_FILE_CHARS * 4:
        raise ValueError(f"File is too large to inspect: {path}")

    content = target.read_text(encoding="utf-8")

    if len(content) > MAX_FILE_CHARS:
        content = content[:MAX_FILE_CHARS] + "\n... [truncated]"

    return content


def search(root: str | Path, query: str) -> list[str]:
    """Search text files for a case-insensitive literal string."""
    if not query:
        raise ValueError("Search query cannot be empty")

    repo_root = Path(root).resolve()
    matches = []

    for file_path in repo_root.rglob("*"):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(repo_root)

        if any(part in IGNORED_DIRS for part in relative.parts):
            continue

        try:
            if file_path.stat().st_size > MAX_FILE_CHARS * 4:
                continue

            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            if query.casefold() in line.casefold():
                matches.append(
                    f"{relative}:{line_number}: {line.strip()[:300]}"
                )

    return matches[:100]


def replace_in_file(
    root: str | Path,
    path: str,
    old: str,
    new: str,
) -> str:
    """Replace one exact, unique text occurrence in a repository file."""
    if not old:
        raise ValueError("The old text cannot be empty")

    target = _safe_path(root, path)

    if not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if target.stat().st_size > MAX_FILE_CHARS * 4:
        raise ValueError(f"File is too large to edit: {path}")

    content = target.read_text(encoding="utf-8")

    occurrences = content.count(old)

    if occurrences == 0:
        raise ValueError("The specified old text was not found")

    if occurrences > 1:
        raise ValueError(
            f"Replacement is ambiguous: found {occurrences} occurrences"
        )

    updated = content.replace(old, new, 1)

    target.write_text(updated, encoding="utf-8")

    return f"Successfully replaced one occurrence in {path}"


import subprocess
import sys


def run_tests(
    root: str | Path,
    timeout: int = 60,
) -> dict:
    """Run pytest in the repository and return a bounded result summary."""
    if timeout < 1 or timeout > 300:
        raise ValueError("Timeout must be between 1 and 300 seconds")

    repo_root = Path(root).resolve()

    if not repo_root.is_dir():
        raise NotADirectoryError(f"Repository directory not found: {root}")

    command = [sys.executable, "-m", "pytest", "-q", "--tb=short"]

    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-12000:],
            "stderr": result.stderr[-4000:],
            "timed_out": False,
        }

    except subprocess.TimeoutExpired as exc:
        def as_text(value):
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value

        return {
            "passed": False,
            "returncode": None,
            "stdout": as_text(exc.stdout)[-12000:],
            "stderr": as_text(exc.stderr)[-4000:],
            "timed_out": True,
        }
