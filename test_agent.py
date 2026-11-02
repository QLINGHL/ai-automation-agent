"""Tests for agent-loop run statistics: tool-call counts + token usage.

Runs two ways:
    python test_agent.py      # plain assert runner, no dependencies
    pytest test_agent.py      # the same functions are pytest tests

The LLM is faked: no API key, no network. The fake mimics the subset of the
OpenAI response shape that run_llm_agent touches (choices[0].message with
model_dump(), tool_calls, and an optional usage payload).
"""

import json
from pathlib import Path

import agent


# ---------------------------------------------------------------------------
# Fake OpenAI response objects
# ---------------------------------------------------------------------------


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        dumped = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            dumped["tool_calls"] = [
                {
                    "id": c.id,
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments,
                    },
                }
                for c in self.tool_calls
            ]
        if exclude_none and dumped["content"] is None:
            dumped.pop("content")
        return dumped


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message, usage=None):
        self.choices = [_FakeChoice(message)]
        self.usage = usage


class _FakeLLM:
    """Replays a script of messages; records nothing else."""

    available = True
    model = "fake-model"

    def __init__(self, script, usage=None):
        self._script = list(script)
        self._usage = usage
        self.calls = 0

    def chat(self, messages, tool_schemas=None):
        self.calls += 1
        message = self._script.pop(0)
        return _FakeResponse(message, usage=self._usage)


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty_and_english():
    assert agent.estimate_tokens("") == 0
    assert agent.estimate_tokens("abcd") == 1  # 4 chars -> 1 token
    assert agent.estimate_tokens("abcde") == 2  # rounds up, never zero


def test_estimate_tokens_cjk_counts_one_each():
    # 3 CJK chars = 3 tokens, regardless of the /4 rule for Latin text.
    assert agent.estimate_tokens("你好世界"[:3]) == 3


# ---------------------------------------------------------------------------
# AgentRunStats bookkeeping
# ---------------------------------------------------------------------------


def test_stats_counts_tools_and_errors():
    stats = agent.AgentRunStats()
    stats.record_tool("text_stats")
    stats.record_tool("text_stats")
    stats.record_tool("bogus_tool", ok=False)
    assert stats.total_tool_calls == 3
    assert stats.tool_calls == {"text_stats": 2, "bogus_tool": 1}
    assert stats.total_tool_errors == 1
    assert stats.tool_errors == {"bogus_tool": 1}
    summary = stats.summary()
    assert summary["total_tool_calls"] == 3
    assert summary["token_source"] == "none"  # no LLM calls yet


def test_format_run_stats_mentions_counts():
    stats = agent.AgentRunStats()
    stats.steps = 2
    stats.record_tool("text_stats")
    stats.prompt_tokens = 10
    stats.completion_tokens = 5
    rendered = agent.format_run_stats(stats)
    assert "Steps: 2" in rendered
    assert "text_stats: 1" in rendered
    assert "10 prompt + 5 completion = 15" in rendered


# ---------------------------------------------------------------------------
# run_llm_agent with a fake LLM
# ---------------------------------------------------------------------------


def test_run_llm_agent_counts_tool_calls_and_estimated_tokens():
    script = [
        _FakeMessage(
            tool_calls=[
                _FakeToolCall(
                    "text_stats", json.dumps({"text": "hello world foo bar"})
                )
            ]
        ),
        _FakeMessage(content="done"),
    ]
    llm = _FakeLLM(script)
    stats = agent.AgentRunStats()
    result = agent.run_llm_agent("count words", llm, stats=stats)
    assert result == "done"
    assert stats.steps == 2
    assert stats.llm_calls == 2
    assert stats.total_tool_calls == 1
    assert stats.tool_calls == {"text_stats": 1}
    assert stats.total_tool_errors == 0
    assert stats.prompt_tokens > 0
    assert stats.completion_tokens > 0
    assert stats.token_source == "estimated"
    assert stats.summary()["total_tokens"] == stats.total_tokens


def test_run_llm_agent_counts_tool_errors_and_keeps_going():
    script = [
        _FakeMessage(
            tool_calls=[
                _FakeToolCall("no_such_tool", json.dumps({})),
            ]
        ),
        _FakeMessage(content="recovered"),
    ]
    llm = _FakeLLM(script)
    stats = agent.AgentRunStats()
    result = agent.run_llm_agent("try a bad tool", llm, stats=stats)
    assert result == "recovered"
    assert stats.tool_calls == {"no_such_tool": 1}
    assert stats.total_tool_errors == 1


def test_run_llm_agent_prefers_real_api_usage():
    script = [
        _FakeMessage(content="done"),
    ]
    # Same usage object on every call: totals must sum across calls.
    llm = _FakeLLM(script, usage=_FakeUsage(100, 20))
    stats = agent.AgentRunStats()
    agent.run_llm_agent("hello", llm, stats=stats)
    assert stats.prompt_tokens == 100
    assert stats.completion_tokens == 20
    assert stats.total_tokens == 120
    assert stats.token_source == "api"


def test_registry_csv_to_markdown_and_schema():
    sample = Path(__file__).parent / "samples" / "sample_sales.csv"
    registry = agent.build_registry(None)
    markdown = registry["csv_to_markdown"](str(sample), preview_rows=2)
    assert markdown.startswith("# CSV Report")
    assert "## Preview (first 2 rows)" in markdown
    assert "| 1003 |" not in markdown
    names = [s["function"]["name"] for s in agent.TOOL_SCHEMAS]
    assert "csv_to_markdown" in names
    assert len(names) == 11


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
