# AI Automation Agent

A Python CLI agent that uses **LLM tool / function calling** to run multi-step
automation workflows over articles, web pages, and CSV data:

**fetch → summarize → extract key points → translate → save a Markdown report**
**analyze CSV → mine keywords from text columns → save a Markdown data report**

Give it a URL or a local text file and it produces a clean Markdown report with
a summary, the key points, and a Chinese translation of the summary. Give it a
CSV file and it produces a data report with dataset statistics, a preview,
and keyword analysis of the free-text column.

## Features

- **Real agent loop** — with an API key configured, the model receives tool
  schemas and decides itself which tools to call and in what order; tool
  results (and errors) are fed back until the task is done, with a step limit
  for safety.
- **Tool registry** — ten focused tools, each a plain Python function:
  `fetch_url`, `read_file`, `save_to_file`, `summarize_text`,
  `extract_key_points`, `translate_text`, `extract_keywords`, `analyze_csv`,
  `text_stats`, `extract_structured`.
- **Data reporting** — `analyze_csv` profiles any CSV with the standard
  library only (shape, inferred column types, min/max/mean, missing-value
  counts, row preview), and `extract_keywords` mines unigram + bigram
  keywords with stopword filtering. Both run fully offline; the `--csv`
  workflow combines them into a Markdown data report.
- **Offline demo mode** — no API key? The same tools run in a deterministic
  pipeline: summarization and key-point extraction fall back to a
  frequency-based extractive algorithm, and translation returns a clearly
  marked placeholder. The full workflow runs end to end with zero network and
  zero cost, so anyone can evaluate it immediately.
- **Any OpenAI-compatible API** — works with OpenAI, Azure OpenAI, or local
  servers (LM Studio, Ollama, vLLM, ...) via `OPENAI_BASE_URL`.
- **Zero-config HTML reading** — web pages are converted to readable text with
  a small built-in extractor (standard library only), no heavy dependencies.

## Tech Stack

- Python 3.10+
- `openai` — chat completions + function calling (lazy-loaded, LLM mode only)
- `requests` — page fetching
- Standard library — HTML parsing, extractive summarization, CSV profiling
  (`csv` module), keyword extraction, `.env` loading

## Project Structure

```
ai-automation-agent/
├── agent.py                  # CLI, agent loop, tool registry + schemas, pipelines
├── tools.py                  # tool implementations + offline fallbacks
├── llm.py                    # minimal OpenAI-compatible client wrapper
├── test_tools.py             # tests for the data & extraction tools
├── requirements.txt
├── .env.example              # copy to .env and add your key
├── samples/
│   ├── sample_article.txt    # bundled article so demo mode needs no network
│   └── sample_sales.csv      # 30-row sample dataset for the data report
└── output/                   # generated reports (git-ignored)
```

## Setup

```bash
git clone <this-repo>
cd ai-automation-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit .env (optional - demo mode needs no key)
```

`.env` values:

| Variable          | Meaning                                   | Default                     |
| ----------------- | ----------------------------------------- | --------------------------- |
| `OPENAI_API_KEY`  | API key; empty = offline demo mode        | *(empty)*                   |
| `OPENAI_BASE_URL` | Any OpenAI-compatible endpoint            | `https://api.openai.com/v1` |
| `MODEL`           | Model name                                | `gpt-4o-mini`               |

## Usage

### 1. Offline demo mode (no API key, no network)

```bash
python agent.py --demo --file samples/sample_article.txt
# -> Saved 2xxx characters to output/report.md
```

### 2. Process a live web page (LLM mode if a key is set, demo pipeline otherwise)

```bash
python agent.py --url https://example.com/some-article --out output/article_report.md
```

### 3. Process a local file with the LLM pipeline

```bash
python agent.py --file path/to/article.txt
```

### 4. Data report from a CSV file (offline-friendly)

```bash
python agent.py --demo --csv samples/sample_sales.csv
# -> Saved 2xxx characters to output/data_report.md
```

The report contains a dataset overview (rows, columns, per-column type,
missing values, min/max/mean for numeric columns), a preview of the first
rows, keywords mined from the free-text column (here `customer_feedback`),
and a summary paragraph. Without `--demo` and with an API key set, the
summary is written by the LLM from the same statistics.

Run the tool tests with:

```bash
python test_tools.py        # or: pytest test_tools.py
```

### 5. Free-form agent loop (requires an API key)

Describe the task in words; the agent plans the tool calls itself:

```bash
python agent.py --task "Fetch https://example.com/some-article, summarize it, translate the summary to Chinese and save everything to output/report.md"
```

Example console output in LLM mode:

```
Running LLM agent loop (model: gpt-4o-mini) ...
  [step 1] tool: fetch_url(url)
  [step 2] tool: summarize_text(text)
  [step 2] tool: extract_key_points(text)
  [step 3] tool: translate_text(text, target_language)
  [step 4] tool: save_to_file(path, content)
Report saved to output/report.md
```

## Sample Output

The generated report (`output/report.md`) looks like:

```markdown
# Automation Report

- **Source:** local file `samples/sample_article.txt`
- **Generated:** 2026-10-20 22:55
- **Mode:** offline demo (extractive fallback, no API key used)

## Summary
...

## Key Points
- ...

## Chinese Translation of the Summary
...

## Source Text (excerpt)
...
```

The data report (`output/data_report.md`, from the bundled
`samples/sample_sales.csv`) looks like:

```markdown
# Data Report

- **Source:** local file `samples/sample_sales.csv`
- **Mode:** offline demo (deterministic analysis, no API key used)

## Dataset Overview

- **Rows:** 30
- **Columns:** 10

| Column | Type | Missing | Min | Max | Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| order_id | integer | 0 | 1001 | 1030 | 1015.5 |
| units | integer | 0 | 1 | 12 | 4.4 |
| revenue | number | 0 | 45 | 779.94 | 262.6683 |
| customer_rating | number | 3 | 3.5 | 5 | 4.4593 |
| customer_feedback | string | 0 | — | — | — |

## Preview (first 5 rows)
...

## Keywords from `customer_feedback`

- quality (15)
- delivery (12)
- fast (9)
- great (9)
- battery (6)

## Summary

The dataset in samples/sample_sales.csv contains 30 rows and 10 columns.
Column revenue ranges from 45 to 779.94 with a mean of 262.6683. ...
```

## How It Works

1. **Tool layer (`tools.py`)** — each tool does one small thing and returns
   predictable text. LLM-backed tools accept an optional client and degrade
   gracefully to deterministic offline implementations.
2. **Agent loop (`agent.py`)** — messages + tool schemas are sent to the
   model; every returned tool call is executed locally and its result is
   appended as a `tool` message. Errors are returned to the model as text so
   it can recover instead of crashing the run.
3. **Demo pipeline (`agent.py`)** — with no key (or `--demo`), the exact same
   tools run in a fixed order, producing the same report structure. This makes
   the project testable in CI and reviewable without any API spend.

## Extending

Add a tool in three steps: write the function in `tools.py`, register it in
`build_registry()`, and add its JSON schema to `TOOL_SCHEMAS` in `agent.py`.
The agent loop and the CLI pick it up automatically.

## License

MIT
