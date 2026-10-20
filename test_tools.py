"""Tests for the data tools: analyze_csv and extract_keywords.

Runs two ways:
    python test_tools.py      # plain assert runner, no dependencies
    pytest test_tools.py      # the same functions are pytest tests
"""

from pathlib import Path

import tools

SAMPLE_CSV = Path(__file__).parent / "samples" / "sample_sales.csv"

KEYWORD_PARAGRAPH = (
    "Automation is changing office work. Teams use automation to process "
    "invoices, and automation tools now handle scheduling and reporting. "
    "With automation, a small team can automate routine approvals and "
    "focus on customers. Modern automation platforms log every step, so "
    "managers trust automation results."
)


def _columns_by_name(analysis: dict) -> dict:
    return {c["name"]: c for c in analysis["columns"]}


def test_analyze_csv_shape_and_types():
    analysis = tools.analyze_csv(str(SAMPLE_CSV))
    assert analysis["row_count"] == 30
    assert analysis["column_count"] == 10
    columns = _columns_by_name(analysis)
    assert columns["order_id"]["type"] == "integer"
    assert columns["units"]["type"] == "integer"
    assert columns["revenue"]["type"] == "number"
    assert columns["product"]["type"] == "string"


def test_analyze_csv_known_stats():
    analysis = tools.analyze_csv(str(SAMPLE_CSV))
    columns = _columns_by_name(analysis)
    # units are 1..12 by design, mean exactly 4.4 over the 30 sample orders
    assert columns["units"]["min"] == 1.0
    assert columns["units"]["max"] == 12.0
    assert columns["units"]["mean"] == 4.4
    # revenue = units * unit_price, verified mean
    assert abs(columns["revenue"]["mean"] - 262.6683) < 0.001
    # exactly three orders in the sample have no customer_rating
    assert columns["customer_rating"]["missing"] == 3
    assert columns["customer_rating"]["mean"] == 4.4593


def test_analyze_csv_preview():
    analysis = tools.analyze_csv(str(SAMPLE_CSV))
    assert len(analysis["preview"]) == 5
    first = analysis["preview"][0]
    assert first["order_id"] == "1001"
    assert first["product"] == "Wireless Mouse"


def test_extract_keywords_top_term():
    keywords = tools.extract_keywords(KEYWORD_PARAGRAPH, top_n=10)
    assert keywords, "expected at least one keyword"
    term, count = keywords[0]
    assert term == "automation"
    assert count == 6
    terms = [t for t, _ in keywords]
    assert "the" not in terms  # stopwords are filtered out


def test_extract_keywords_includes_bigrams():
    text = (
        "Machine learning powers the report. Machine learning models rank "
        "the rows, and machine learning improves every quarter."
    )
    keywords = dict(tools.extract_keywords(text, top_n=10))
    assert keywords.get("machine learning", 0) >= 3


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
