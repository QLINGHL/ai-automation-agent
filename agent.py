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
    "and save a well-structured Markdown report with save_to_file. When the "
    "report is saved, reply with a one-line confirmation."
)


# ---------------------------------------------------------------------------
# LLM agent loop
# ---------------------------------------------------------------------------


def run_llm_agent(task: str, llm: LLMClient, max_steps: int = 12) -> str:
    registry = build_registry(llm)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for step in range(1, max_steps + 1):
        response = llm.chat(messages, TOOL_SCHEMAS)
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
            except Exception as exc:  # feed errors back so the model can recover
                result = f"ERROR in {name}: {exc}"
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
        "--task",
        help="Free-form task for the LLM agent loop (requires an API key)",
    )
    parser.add_argument(
        "--out",
        default="output/report.md",
        help="Where to save the Markdown report (default: output/report.md)",
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
            print(run_llm_agent(args.task, llm))
            return 0

        source = args.url or args.file
        if not source:
            parser.error("one of --url, --file or --task is required")

        if not llm.available and not args.demo:
            print("No OPENAI_API_KEY found -> running in offline demo mode.")
        run_pipeline(source, args.out, llm, demo=args.demo)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
