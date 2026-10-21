"""Tool implementations for the AI Automation Agent.

Every tool is a plain Python function. The same functions are:

* exposed to the LLM through JSON schemas (see TOOL_SCHEMAS in agent.py), and
* called directly by the deterministic offline pipeline in demo mode.

Tools that conceptually need an LLM (summarize / key points / translate)
accept an optional ``llm`` argument. When no LLM is available they fall back
to deterministic offline implementations, so the full workflow always runs
end to end -- with or without an API key. The data tools (``analyze_csv``,
``extract_keywords``, ``text_stats``) are pure standard library and work
fully offline.
"""

import csv
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


def extract_keywords(text: str, top_n: int = 10, llm=None) -> list[tuple[str, int]]:
    """Extract the top keywords/phrases of a text as (term, count) pairs.

    Fully offline: unigrams and bigrams (consecutive content words) are
    counted with stopwords filtered out, then ranked by count with
    first-occurrence order as the tie-breaker. When an LLM is configured it
    may refine the selection, but it can only reorder/trim the frequency
    candidates -- the offline list is always the fallback.
    """
    sentences = re.split(r"[.!?\n]+", text.lower())
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    first_seen: dict[str, int] = {}
    position = 0

    def _note(term: str) -> None:
        nonlocal position
        if term not in first_seen:
            first_seen[term] = position
        position += 1

    for sentence in sentences:
        tokens = [
            w
            for w in re.findall(r"[a-zA-Z']+", sentence)
            if w not in _STOPWORDS and len(w) > 2
        ]
        for token in tokens:
            unigrams[token] += 1
            _note(token)
        for first, second in zip(tokens, tokens[1:]):
            phrase = f"{first} {second}"
            bigrams[phrase] += 1
            _note(phrase)

    counts = dict(unigrams)
    counts.update(bigrams)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    result = ranked[:top_n]

    if llm is not None and llm.available and result:
        try:
            candidates = ", ".join(term for term, _ in ranked[: top_n * 2])
            raw = llm.complete(
                "From the candidate keywords below, pick the up to "
                f"{top_n} most meaningful ones for the text, most important "
                "first, and return them comma-separated, exactly as written "
                f"in the candidate list.\n\nCandidates: {candidates}\n\n"
                f"Text:\n{text[:4000]}",
                system="You are a precise keyword-extraction assistant.",
            )
            picked = [p.strip().lower() for p in raw.split(",")]
            refined = [(t, c) for t, c in ranked if t in set(picked)]
            refined.sort(key=lambda kv: picked.index(kv[0]))
            if len(refined) >= 3:
                return refined[:top_n]
        except Exception:
            pass  # any LLM failure falls back to the frequency ranking
    return result


# --------------------------------------------------------------------------
# CSV analysis (pure stdlib, works offline)
# --------------------------------------------------------------------------


def _infer_column(values: list[str]) -> tuple[str, list[float] | None]:
    """Infer a column type from its non-missing values.

    Returns (type, numeric_values) where type is "integer", "number",
    "string" or "empty"; numeric_values is set for the two numeric types.
    """
    if not values:
        return "empty", None
    numbers: list[float] = []
    all_int = True
    for value in values:
        try:
            numbers.append(int(value))
        except ValueError:
            all_int = False
            try:
                numbers.append(float(value))
            except ValueError:
                return "string", None
    return ("integer" if all_int else "number"), [float(n) for n in numbers]


def analyze_csv(path: str, preview_rows: int = 5) -> dict:
    """Analyze a CSV file: shape, column types, numeric stats, missing
    values and a preview of the first rows. Pure standard library."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if any(c.strip() for c in row)]
    if not rows:
        return {
            "path": str(path),
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "preview": [],
        }

    header = [name.strip() or f"column_{i + 1}" for i, name in enumerate(rows[0])]
    data = rows[1:]
    width = len(header)

    columns = []
    for idx, name in enumerate(header):
        raw_values = [(row[idx] if idx < len(row) else "") for row in data]
        present = [v.strip() for v in raw_values if v.strip()]
        col_type, numbers = _infer_column(present)
        column: dict = {
            "name": name,
            "type": col_type,
            "missing": len(raw_values) - len(present),
        }
        if numbers:
            column["min"] = min(numbers)
            column["max"] = max(numbers)
            column["mean"] = round(sum(numbers) / len(numbers), 4)
        columns.append(column)

    preview = [
        {header[i]: (row[i] if i < len(row) else "") for i in range(width)}
        for row in data[:preview_rows]
    ]
    return {
        "path": str(path),
        "row_count": len(data),
        "column_count": width,
        "columns": columns,
        "preview": preview,
    }


def text_stats(text: str) -> dict:
    """Compute basic statistics about a text. Pure standard library.

    Returns character/word/sentence/paragraph counts, the average sentence
    length, the longest sentence (in words) and an estimated reading time
    at 200 words per minute. Words are runs of word characters, so a run of
    CJK characters counts as one token -- the counts are meant as a quick
    profile, not a linguistic analysis.
    """
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    # Own sentence split: _split_sentences drops short sentences on purpose
    # (it feeds the summarizer), which would skew these counts.
    flat = re.sub(r"\s+", " ", text).strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()] if flat else []
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    sentence_word_counts = [
        len(re.findall(r"\w+", s, flags=re.UNICODE)) for s in sentences
    ]
    word_count = len(words)
    return {
        "char_count": len(text),
        "char_count_no_spaces": len(re.sub(r"\s", "", text)),
        "word_count": word_count,
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_words_per_sentence": (
            round(word_count / len(sentences), 2) if sentences else 0.0
        ),
        "longest_sentence_words": max(sentence_word_counts, default=0),
        "reading_time_minutes": round(word_count / 200, 2),
    }


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
