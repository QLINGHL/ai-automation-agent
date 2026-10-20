"""Minimal OpenAI-compatible LLM client wrapper.

Reads configuration from environment variables (usually via a .env file):
    OPENAI_API_KEY   - API key (empty = LLM unavailable, demo mode only)
    OPENAI_BASE_URL  - API base URL (any OpenAI-compatible endpoint)
    MODEL            - model name, defaults to gpt-4o-mini

The `openai` package is imported lazily so that the offline demo pipeline
works even in environments where the package is not installed.
"""

import os


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        self.model = os.getenv("MODEL", "gpt-4o-mini").strip()
        self._client = None

    @property
    def available(self) -> bool:
        """True when an API key is configured."""
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import: not needed for demo mode

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def complete(self, prompt: str, system: str = "You are a helpful assistant.") -> str:
        """Single-shot chat completion, returns the message text."""
        resp = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def chat(self, messages: list, tool_schemas: list | None = None):
        """One chat round, optionally with tool schemas. Returns the raw response."""
        kwargs = {"model": self.model, "messages": messages}
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        return self._get_client().chat.completions.create(**kwargs)
