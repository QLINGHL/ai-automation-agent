#!/usr/bin/env python3
"""AI Automation Agent - a CLI agent built on LLM tool/function calling.

Given a URL or a local article file, the agent runs a multi-step workflow:

    fetch/read -> summarize -> extract key points -> translate -> save report

Two execution modes:

* **LLM mode** (OPENAI_API_KEY set): an agent loop sends the task plus tool
  schemas to the model; the model decides which tools to call and in what
  order, tool results are fed back, and the loop runs until the task is done.
* **Demo mode** (no API key, or --demo): the same tools run in a fixed,
  deterministic offline pipeline. Summarization / key points fall back to a
  frequency-based extractive algorithm and translation returns a clearly
  marked placeholder, so the whole workflow runs end to end with no network
  and no API key -- ideal for a quick evaluation.

Usage:
    python agent.py --demo --file samples/sample_article.txt
    python agent.py --demo --csv samples/sample_sales.csv
    python agent.py --url https://example.com/some-article
    python agent.py --task "Fetch https://example.com/a, summarize it, \
        translate the summary to Chinese and save everything to output/r.md"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import tools
from llm import LLMClient

# ---------------------------------------------------------------------------
# .env loading (tiny parser, no extra dependency)
# ---------------------------------------------------------------------------


def load_dotenv(path: str = ".env") -> None:
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Tool registry + OpenAI function-calling schemas
# ---------------------------------------------------------------------------


def build_registry(llm: LLMClient) -> dict:
    """Map tool name -> callable. LLM-aware tools close over the client."""
    return {
        "fetch_url": lambda url: tools.fetch_url(url),
        "read_file": lambda path: tools.read_file(path),
        "save_to_file": lambda path, content: tools.save_to_file(path, content),
        "summarize_text": lambda text: tools.summarize_text(text, llm=llm),
        "extract_key_points": lambda text: "\n".join(
            f"- {p}" for p in tools.extract_key_points(text, llm=llm)
        ),
        "translate_text": lambda text, target_language="Chinese": tools.translate_text(
            text, target_language, llm=llm
        ),
        "extract_keywords": lambda text, top_n=10: "\n".join(
            f"- {term} ({count})"
            for term, count in tools.extract_keywords(text, top_n, llm=llm)
        ),
        "analyze_csv": lambda path: json.dumps(
            tools.analyze_csv(path), indent=2, ensure_ascii=False
        ),
        "csv_to_markdown": lambda path, preview_rows=10: tools.csv_to_markdown(
            path, preview_rows
        ),
        "text_stats": lambda text: json.dumps(
            tools.text_stats(text), indent=2, ensure_ascii=False
        ),
        "extract_structured": lambda text, schema: json.dumps(
            tools.extract_structured(text, schema, llm=llm),
            indent=2,
            ensure_ascii=False,
        ),
    }


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Download a web page and return its readable plain text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a local UTF-8 text file and return its content.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_file",
            "description": "Save text content to a local file, creating folders as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Summarize a long text into one concise paragraph.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_key_points",
            "description": "Extract the key points of a text as a bullet list.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_keywords",
            "description": "Extract the top keywords and phrases of a text "
            "with their frequencies (works offline).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "top_n": {"type": "integer", "default": 10},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_csv",
            "description": "Analyze a CSV file: row/column counts, column "
            "names and inferred types, numeric stats (min/max/mean), "
            "missing-value counts and a preview of the first rows.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "csv_to_markdown",
            "description": "Render a CSV file as a Markdown report: dataset "
            "overview table (column types, missing values, numeric stats) "
            "plus a preview table, ready to paste into a report or save "
            "with save_to_file (works offline).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "preview_rows": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_stats",
            "description": "Compute statistics about a text: character, "
            "word, sentence and paragraph counts, average sentence length "
            "and estimated reading time (works offline).",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_structured",
            "description": "Extract structured fields from a text according "
            "to a JSON Schema; returns a JSON object with null for fields "
            "not found. LLM mode when an API key is set, deterministic "
            "offline regex fallback otherwise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "schema": {
                        "type": "object",
                        "description": "JSON Schema object; top-level "
                        "'properties' are extracted (with optional 'type', "
                        "'format', 'enum' or 'pattern' hints).",
                    },
                },
                "required": ["text", "schema"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text into the target language (default Chinese).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_language": {"type": "string", "default": "Chinese"},
                },
                "required": ["text"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are an automation agent with access to tools. Complete the user's "
    "task step by step: fetch or read the source material first, then "
    "summarize it, extract key points, translate the summary to Chinese, "
    "and save a well-structured Markdown report with save_to_file. For CSV "
    "data, use analyze_csv first and extract_keywords on any free-text "
    "column, then save a data report the same way. When the report is "
    "saved, reply with a one-line confirmation."
)


# ---------------------------------------------------------------------------
# Agent run statistics (tool-call counts + token usage)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Offline token estimate for a piece of text (no API call needed).

    Heuristic, documented so runs stay comparable: each CJK character counts
    as one token, other characters count as a quarter token (rounded up).
    Real API usage, when the model response carries it, always wins over
    this estimate -- see :meth:`AgentRunStats.record_llm_call`.
    """
    if not text:
        return 0
    import math

    cjk = sum(
        1
        for ch in text
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7a3"
    )
    return cjk + math.ceil((len(text) - cjk) / 4)


class AgentRunStats:
    """Counters for one agent-loop run: tool calls and token usage.

    Pass an instance to :func:`run_llm_agent` and read it back afterwards;
    the loop fills it in place, so the loop's string return value is
    unchanged and existing callers keep working.
    """

    def __init__(self) -> None:
        self.steps = 0
        self.llm_calls = 0
        self.tool_calls: dict[str, int] = {}
        self.tool_errors: dict[str, int] = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_usage_calls = 0

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())

    @property
    def total_tool_errors(self) -> int:
        return sum(self.tool_errors.values())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def token_source(self) -> str:
        if self.llm_calls == 0:
            return "none"
        if self.api_usage_calls == self.llm_calls:
            return "api"
        if self.api_usage_calls == 0:
            return "estimated"
        return "mixed"

    def record_tool(self, name: str, ok: bool = True) -> None:
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        if not ok:
            self.tool_errors[name] = self.tool_errors.get(name, 0) + 1

    def record_llm_call(self, response, prompt_messages=None) -> None:
        """Count one LLM round, preferring real usage when present."""
        self.llm_calls += 1
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        prompt = completion = None
        if usage is not None:
            if isinstance(usage, dict):
                prompt = usage.get("prompt_tokens")
                completion = usage.get("completion_tokens")
                if prompt is None and completion is None:
                    total = usage.get("total_tokens")
                    if total is not None:
                        prompt, completion = total, 0
            else:
                prompt = getattr(usage, "prompt_tokens", None)
                completion = getattr(usage, "completion_tokens", None)
                if prompt is None and completion is None:
                    total = getattr(usage, "total_tokens", None)
                    if total is not None:
                        prompt, completion = total, 0
        if prompt is not None or completion is not None:
            self.api_usage_calls += 1
            self.prompt_tokens += int(prompt or 0)
            self.completion_tokens += int(completion or 0)
            return
        # No usage on the response (offline fakes, some providers): estimate
        # from the prompt messages and the returned message content / calls.
        prompt_text = json.dumps(
            prompt_messages or [], ensure_ascii=False, default=str
        )
        message = None
        if not isinstance(response, dict):
            choices = getattr(response, "choices", None)
            if choices:
                message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) or ""
        calls = getattr(message, "tool_calls", None) or []
        completion_text = content + json.dumps(
            [
                {"name": c.function.name, "arguments": c.function.arguments}
                for c in calls
            ],
            ensure_ascii=False,
            default=str,
        )
        self.prompt_tokens += estimate_tokens(prompt_text)
        self.completion_tokens += estimate_tokens(completion_text)

    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "llm_calls": self.llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "tool_calls_by_tool": dict(self.tool_calls),
            "total_tool_errors": self.total_tool_errors,
            "tool_errors_by_tool": dict(self.tool_errors),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "token_source": self.token_source,
        }


def format_run_stats(stats: AgentRunStats) -> str:
    """Render a run's statistics as a short Markdown section."""
    lines = [
        "## Run Statistics",
        "",
        f"- Steps: {stats.steps}",
        f"- LLM calls: {stats.llm_calls}",
        f"- Tool calls: {stats.total_tool_calls}",
    ]
    for name in sorted(stats.tool_calls):
        lines.append(f"  - {name}: {stats.tool_calls[name]}")
    lines.append(f"- Tool errors: {stats.total_tool_errors}")
    for name in sorted(stats.tool_errors):
        lines.append(f"  - {name}: {stats.tool_errors[name]} errors")
    lines.append(
        f"- Tokens: {stats.prompt_tokens} prompt + "
        f"{stats.completion_tokens} completion = {stats.total_tokens} "
        f"({stats.token_source})"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM agent loop
# ---------------------------------------------------------------------------


def run_llm_agent(
    task: str,
    llm: LLMClient,
    max_steps: int = 12,
    stats: AgentRunStats | None = None,
) -> str:
    if stats is None:
        stats = AgentRunStats()
    registry = build_registry(llm)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for step in range(1, max_steps + 1):
        stats.steps = step
        response = llm.chat(messages, TOOL_SCHEMAS)
        stats.record_llm_call(response, prompt_messages=messages)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content or "(agent finished without a message)"

        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                print(f"  [step {step}] tool: {name}({', '.join(args)})")
                result = registry[name](**args)
                stats.record_tool(name, ok=True)
            except Exception as exc:  # feed errors back so the model can recover
                result = f"ERROR in {name}: {exc}"
                stats.record_tool(name, ok=False)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result)[: tools.MAX_TEXT_CHARS],
                }
            )
    return "Stopped: max steps reached before the task completed."


# ---------------------------------------------------------------------------
# Deterministic offline pipeline (demo mode)
# ---------------------------------------------------------------------------


def build_report(source_label: str, article: str, llm: LLMClient, mode: str) -> str:
    summary = tools.summarize_text(article, llm=llm)
    key_points = tools.extract_key_points(article, llm=llm)
    translation = tools.translate_text(summary, "Chinese", llm=llm)

    lines = [
        "# Automation Report",
        "",
        f"- **Source:** {source_label}",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- **Mode:** {mode}",
        "",
        "## Summary",
        "",
        summary,
        "",
        "## Key Points",
        "",
    ]
    lines += [f"- {point}" for point in key_points]
    lines += [
        "",
        "## Chinese Translation of the Summary",
        "",
        translation,
        "",
        "## Source Text (excerpt)",
        "",
        article[:800].strip() + ("..." if len(article) > 800 else ""),
        "",
    ]
    return "\n".join(lines)


def run_pipeline(source: str, out_path: str, llm: LLMClient, demo: bool) -> str:
    if source.startswith(("http://", "https://")):
        print(f"Fetching {source} ...")
        article = tools.fetch_url(source)
        label = source
    else:
        print(f"Reading {source} ...")
        article = tools.read_file(source)
        label = f"local file `{source}`"

    mode = (
        "offline demo (extractive fallback, no API key used)"
        if demo or not llm.available
        else f"LLM ({llm.model})"
    )
    report = build_report(label, article, None if demo else llm, mode)
    status = tools.save_to_file(out_path, report)
    print(status)
    return out_path


# ---------------------------------------------------------------------------
# Data report pipeline (CSV -> Markdown data report)
# ---------------------------------------------------------------------------


def _md_cell(value) -> str:
    return str(value).replace("|", "\\|").strip()


def _fmt_stat(value: float) -> str:
    """Render a stat without a trailing .0 for integral values."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _pick_text_column(analysis: dict, rows_text: dict[str, list[str]]) -> str | None:
    """Pick the free-text column (longest average values) for keyword mining."""
    best_name, best_avg = None, 0.0
    for column in analysis["columns"]:
        if column["type"] != "string":
            continue
        values = [v for v in rows_text.get(column["name"], []) if v.strip()]
        if len(values) < 3:
            continue
        avg = sum(len(v) for v in values) / len(values)
        if avg > best_avg:
            best_name, best_avg = column["name"], avg
    return best_name if best_avg >= 15 else None


def build_data_report(csv_path: str, llm: LLMClient, mode: str) -> str:
    import csv as csv_module

    analysis = tools.analyze_csv(csv_path)
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        records = list(csv_module.DictReader(handle))
    rows_text: dict[str, list[str]] = {}
    for record in records:
        for name, value in record.items():
            if name:
                rows_text.setdefault(name, []).append(value or "")

    lines = [
        "# Data Report",
        "",
        f"- **Source:** local file `{csv_path}`",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- **Mode:** {mode}",
        "",
        "## Dataset Overview",
        "",
        f"- **Rows:** {analysis['row_count']}",
        f"- **Columns:** {analysis['column_count']}",
        "",
        "| Column | Type | Missing | Min | Max | Mean |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for column in analysis["columns"]:
        stats = [
            _fmt_stat(column[key]) if key in column else "—"
            for key in ("min", "max", "mean")
        ]
        lines.append(
            f"| {_md_cell(column['name'])} | {column['type']} | "
            f"{column['missing']} | {stats[0]} | {stats[1]} | {stats[2]} |"
        )

    names = [c["name"] for c in analysis["columns"]]
    if analysis["preview"]:
        lines += [
            "",
            f"## Preview (first {len(analysis['preview'])} rows)",
            "",
            "| " + " | ".join(_md_cell(n) for n in names) + " |",
            "| " + " | ".join("---" for _ in names) + " |",
        ]
        for row in analysis["preview"]:
            lines.append(
                "| " + " | ".join(_md_cell(row.get(n, "")) for n in names) + " |"
            )

    text_column = _pick_text_column(analysis, rows_text)
    keywords: list[tuple[str, int]] = []
    if text_column:
        joined = " ".join(rows_text[text_column])
        keywords = tools.extract_keywords(joined, 10, llm=llm)
        lines += ["", f"## Keywords from `{text_column}`", ""]
        lines += [f"- {term} ({count})" for term, count in keywords]

    numeric = [c for c in analysis["columns"] if "mean" in c]
    digest_bits = [
        f"The dataset in {csv_path} contains {analysis['row_count']} rows "
        f"and {analysis['column_count']} columns."
    ]
    for column in numeric:
        digest_bits.append(
            f"Column {column['name']} ranges from {_fmt_stat(column['min'])} "
            f"to {_fmt_stat(column['max'])} with a mean of "
            f"{_fmt_stat(column['mean'])}."
        )
    if keywords:
        digest_bits.append(
            "The most frequent terms in the "
            f"{text_column} column are "
            + ", ".join(term for term, _ in keywords[:5])
            + "."
        )
    digest = " ".join(digest_bits)
    if llm is not None and llm.available:
        summary = llm.complete(
            "Write a short summary paragraph (3-5 sentences) of this dataset "
            "digest for a business reader. Keep the numbers exact.\n\n" + digest,
            system="You are a precise data-reporting assistant.",
        )
    else:
        summary = digest
    lines += ["", "## Summary", "", summary, ""]
    return "\n".join(lines)


def run_data_pipeline(csv_path: str, out_path: str, llm: LLMClient, demo: bool) -> str:
    print(f"Analyzing {csv_path} ...")
    mode = (
        "offline demo (deterministic analysis, no API key used)"
        if demo or not llm.available
        else f"LLM ({llm.model})"
    )
    report = build_data_report(csv_path, None if demo else llm, mode)
    status = tools.save_to_file(out_path, report)
    print(status)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI Automation Agent: fetch -> summarize -> key points "
        "-> translate -> Markdown report."
    )
    parser.add_argument("--url", help="Article URL to process")
    parser.add_argument("--file", help="Local article text file to process")
    parser.add_argument(
        "--csv",
        help="Local CSV file to analyze into a Markdown data report",
    )
    parser.add_argument(
        "--task",
        help="Free-form task for the LLM agent loop (requires an API key)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to save the report (default: output/report.md, or "
        "output/data_report.md with --csv)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force offline demo mode (no API calls, deterministic fallbacks)",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    llm = LLMClient()

    try:
        if args.task:
            if not llm.available:
                print(
                    "ERROR: --task needs OPENAI_API_KEY in .env "
                    "(free-form agent loop has no offline mode).",
                    file=sys.stderr,
                )
                return 2
            print(f"Running LLM agent loop (model: {llm.model}) ...")
            stats = AgentRunStats()
            print(run_llm_agent(args.task, llm, stats=stats))
            print()
            print(format_run_stats(stats))
            return 0

        if not llm.available and not args.demo and (args.url or args.file or args.csv):
            print("No OPENAI_API_KEY found -> running in offline demo mode.")

        if args.csv:
            run_data_pipeline(
                args.csv, args.out or "output/data_report.md", llm, demo=args.demo
            )
            return 0

        source = args.url or args.file
        if not source:
            parser.error("one of --url, --file, --csv or --task is required")

        run_pipeline(source, args.out or "output/report.md", llm, demo=args.demo)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
