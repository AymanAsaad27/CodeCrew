import time
from typing import TypeVar

import ollama
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    """Raised when the model-call budget is exhausted."""


class LLM:
    """Ollama client with structured output, retries, and usage tracking."""

    def __init__(
        self,
        max_calls: int = 30,
        num_ctx: int = 4096,
        timeout: float = 180.0,
    ):
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")

        self.max_calls = max_calls
        self.num_ctx = num_ctx
        self.timeout = timeout

        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_seconds = 0.0

        self.client = ollama.Client(
    host="http://127.0.0.1:11434",
    timeout=timeout,
    trust_env=False,
)

    def json(
        self,
        model: str,
        messages: list[dict],
        schema: type[T],
        temperature: float = 0.0,
        retries: int = 2,
    ) -> T:
        """Return a validated Pydantic object from an Ollama response."""

        if retries < 0:
            raise ValueError("retries cannot be negative")

        current_messages = list(messages)
        last_error = None

        for attempt in range(retries + 1):
            if self.calls >= self.max_calls:
                raise BudgetExceeded(
                    f"Model-call budget exhausted ({self.max_calls} calls)"
                )

            self.calls += 1
            started = time.monotonic()

            try:
                response = self.client.chat(
                    model=model,
                    messages=current_messages,
                    format=schema.model_json_schema(),
                    options={
                        "temperature": temperature,
                        "num_ctx": self.num_ctx,
                    },
                )
            finally:
                self.llm_seconds += time.monotonic() - started

            self.prompt_tokens += getattr(response, "prompt_eval_count", 0) or 0
            self.completion_tokens += getattr(response, "eval_count", 0) or 0

            content = response.message.content or ""

            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                last_error = exc

                if attempt < retries:
                    current_messages = current_messages + [
                        {
                            "role": "assistant",
                            "content": content,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Your previous response did not satisfy the "
                                f"required JSON schema. Validation error: {exc}. "
                                "Return a corrected JSON response only."
                            ),
                        },
                    ]

        raise ValueError(
            f"Model did not produce valid structured output: {last_error}"
        )
