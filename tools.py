"""Tool implementations for the AI Automation Agent.

Every tool is a plain Python function. The same functions are:

* exposed to the LLM through JSON schemas (see TOOL_SCHEMAS in agent.py), and
* called directly by the deterministic offline pipeline in demo mode.

Tools that conceptually need an LLM (summarize / key points / translate)
accept an optional ``llm`` argument. When no LLM is available they fall back
to deterministic offline implementations, so the full workflow always runs
end to end -- with or without an API key.
"""

import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import requests

MAX_TEXT_CHARS = 12_000  # cap text sent to the LLM / kept in reports

# --------------------------------------------------------------------------
# Web fetching
# --------------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """Very small HTML -> text extractor (skips script/style/noscript)."""

    SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            text = data.strip()
            if text:
                self.parts.append(text)


def fetch_url(url: str, timeout: int = 15) -> str:
    """Download a web page and return its readable plain text."""
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "ai-automation-agent/1.0 (+portfolio demo)"},
    )
    resp.raise_for_status()
    parser = _TextExtractor()
    parser.feed(resp.text)
    text = "\n".join(parser.parts)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --------------------------------------------------------------------------
# File I/O
# --------------------------------------------------------------------------


def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its content."""
    return Path(path).read_text(encoding="utf-8")


def save_to_file(path: str, content: str) -> str:
    """Write content to a file (creating parent dirs). Returns a status line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Saved {len(content)} characters to {target}"


# --------------------------------------------------------------------------
# Offline (deterministic) text processing helpers
# --------------------------------------------------------------------------

_STOPWORDS = set(
    "the a an and or but if while of at by for with about to from in on is are "
    "was were be been being have has had do does did will would can could "
    "should may might must shall this that these those it its as not no yes "
    "than then so such into over under more most other some any only own same "
    "too very just also".split()
)


def _split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", flat)
    return [s.strip() for s in sentences if len(s.strip()) > 25]


def _score_sentences(sentences: list[str]) -> list[tuple[float, int, str]]:
    words = re.findall(r"[a-zA-Z']+", " ".join(sentences).lower())
    freq = Counter(w for w in words if w not in _STOPWORDS and len(w) > 2)
    scored = []
    for idx, sentence in enumerate(sentences):
        tokens = re.findall(r"[a-zA-Z']+", sentence.lower())
        if not tokens:
            continue
        score = sum(freq[t] for t in tokens) / len(tokens)
        scored.append((score, idx, sentence))
    return scored


def _extractive_summary(text: str, max_sentences: int = 5) -> str:
    """Frequency-based extractive summary (classic Luhn-style scoring)."""
    sentences = _split_sentences(text)
    if not sentences:
        return text.strip()[:500]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    scored = _score_sentences(sentences)
    top = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(s for _, _, s in sorted(top, key=lambda item: item[1]))


def _extractive_key_points(text: str, max_points: int = 5) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    scored = _score_sentences(sentences)
    top = sorted(scored, reverse=True)[:max_points]
    return [s for _, _, s in sorted(top, key=lambda item: item[1])]


# --------------------------------------------------------------------------
# LLM-backed tools (with offline fallbacks)
# --------------------------------------------------------------------------


def summarize_text(text: str, llm=None) -> str:
    """Summarize text. Uses the LLM when available, else extractive summary."""
    text = text[:MAX_TEXT_CHARS]
    if llm is not None and llm.available:
        return llm.complete(
            "Summarize the following text in one concise paragraph "
            "(about 120-180 words). Keep the key facts and numbers.\n\n" + text,
            system="You are a precise summarization assistant.",
        )
    return _extractive_summary(text)


def extract_key_points(text: str, llm=None) -> list[str]:
    """Extract key points as a list. LLM when available, else top sentences."""
    text = text[:MAX_TEXT_CHARS]
    if llm is not None and llm.available:
        raw = llm.complete(
            "Extract the 5 most important key points from the following text. "
            "Return one point per line, each line starting with '- '.\n\n" + text,
            system="You are a precise information-extraction assistant.",
        )
        points = [line[2:].strip() for line in raw.splitlines() if line.strip().startswith("-")]
        if points:
            return points
    return _extractive_key_points(text)


def translate_text(text: str, target_language: str = "Chinese", llm=None) -> str:
    """Translate text. Uses the LLM when available; offline returns a marked
    placeholder note (translation genuinely requires a model or MT service)."""
    text = text[:MAX_TEXT_CHARS]
    if llm is not None and llm.available:
        return llm.complete(
            f"Translate the following text into {target_language}. "
            "Output only the translation.\n\n" + text,
            system="You are a professional translator.",
        )
    return (
        f"[DEMO MODE - offline placeholder] Translation to {target_language} is "
        "not available without an LLM API key, so no machine translation was "
        "performed. Configure OPENAI_API_KEY in .env and re-run without "
        "--demo to get a real translation of the summary."
    )
