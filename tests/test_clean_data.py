#!/usr/bin/env python3
"""Offline checks for analysis/clean_data.py.

Every test builds its own in-memory CSV -- the real dataset is never needed, so
the suite runs in a fresh checkout.

Run with:
    python tests/test_clean_data.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import clean_data as cd  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

#: A comma inside a quoted title is the exact case a naive `split(",")` breaks
#: on, so it is present in row 1 on purpose.
RAW_CSV = """title,url,published_date,author,tags,excerpt,scraped_at
"Gemini 3.7 Flash: Features, Benchmarks, and Pricing",https://x.test/blog/gemini,2026-08-14,Matt,Artificial Intelligence|Large Language Models,"Google's model, benchmarked.",2026-08-16
Positional Encoding Explained,https://x.test/blog/positional,2026-08-14,Tim,Deep Learning,How transformers track order.,2026-08-16
An Untagged Post,https://x.test/blog/untagged,2025-01-02,Ann,,Still a real article.,2026-08-16
Positional Encoding Explained,https://x.test/blog/positional,2026-08-14,Tim,Deep Learning,Duplicate row.,2026-08-16
SQL Certification Course | DataCamp,https://x.test/blog/sql-certification,,,,Earn a SQL Associate certification.,2026-08-16
,https://x.test/blog/no-title,2024-05-05,Bob,SQL,Body text.,2026-08-16
No Excerpt Here,https://x.test/blog/no-excerpt,2024-05-06,Bob,SQL,,2026-08-16
Broken Date,https://x.test/blog/broken-date,14 August 2026,Bob,SQL,Body text.,2026-08-16
"""


def write_raw(directory: Path, content: str = RAW_CSV) -> Path:
    """Write a raw CSV fixture to disk.

    Args:
        directory: Temporary directory.
        content: CSV text.

    Returns:
        Path to the written file.
    """
    path = directory / "articles_raw.csv"
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_load_raw_handles_quoted_commas() -> None:
    """A comma inside a quoted title must not shift the columns."""
    with tempfile.TemporaryDirectory() as tmp:
        frame = cd.load_raw(write_raw(Path(tmp)))

    assert list(frame.columns) == list(cd.REQUIRED_COLUMNS)
    assert len(frame) == 8
    row = frame.iloc[0]
    assert row["title"] == "Gemini 3.7 Flash: Features, Benchmarks, and Pricing"
    assert row["url"] == "https://x.test/blog/gemini"
    assert row["published_date"] == "2026-08-14"


def test_load_raw_keeps_empty_strings_not_nan() -> None:
    """Missing fields stay as ``""`` so string operations keep working."""
    with tempfile.TemporaryDirectory() as tmp:
        frame = cd.load_raw(write_raw(Path(tmp)))

    assert not frame.isna().any().any()
    certification = frame[frame["url"].str.endswith("sql-certification")].iloc[0]
    assert certification["published_date"] == ""
    assert certification["tags"] == ""


def test_load_raw_rejects_missing_columns() -> None:
    """A file without the expected schema fails loudly."""
    with tempfile.TemporaryDirectory() as tmp:
        path = write_raw(Path(tmp), "title,url\nA,https://x.test/blog/a\n")
        try:
            cd.load_raw(path)
        except cd.SchemaError as exc:
            assert "published_date" in str(exc)
        else:
            raise AssertionError("Expected SchemaError")


def test_load_raw_missing_file() -> None:
    """A missing input file raises FileNotFoundError, not a pandas error."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            cd.load_raw(Path(tmp) / "nope.csv")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Expected FileNotFoundError")


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def test_clean_drops_every_bad_row_once() -> None:
    """Each fixture defect is removed exactly once, with its own reason."""
    with tempfile.TemporaryDirectory() as tmp:
        frame, dropped, report = cd.clean(cd.load_raw(write_raw(Path(tmp))))

    assert report.rows_in == 8
    assert report.rows_out == 3
    assert report.rows_dropped == 5
    assert report.dropped == {
        "missing published_date (certification page)": 1,
        "unparsable published_date": 1,
        "empty title": 1,
        "empty excerpt": 1,
        "duplicate url": 1,
    }

    kept = set(frame["url"])
    assert kept == {
        "https://x.test/blog/gemini",
        "https://x.test/blog/positional",
        "https://x.test/blog/untagged",
    }
    assert len(dropped) == 5
    assert set(dropped[cd.DROP_REASON_COLUMN]) == set(report.dropped)


def test_certification_pages_are_removed() -> None:
    """The empty-date rule is what strips the certification landing pages."""
    with tempfile.TemporaryDirectory() as tmp:
        frame, dropped, _ = cd.clean(cd.load_raw(write_raw(Path(tmp))))

    assert not frame["url"].str.contains("certification").any()
    reason = dropped.loc[dropped["url"].str.contains("certification"), cd.DROP_REASON_COLUMN]
    assert reason.iloc[0] == "missing published_date (certification page)"


def test_duplicate_urls_keep_the_first_row() -> None:
    """Deduplication keeps the first occurrence and drops the rest."""
    with tempfile.TemporaryDirectory() as tmp:
        frame, _, _ = cd.clean(cd.load_raw(write_raw(Path(tmp))))

    assert frame["url"].is_unique
    positional = frame[frame["url"].str.endswith("positional")].iloc[0]
    assert positional["excerpt"] == "How transformers track order."


def test_dates_are_parsed_and_derived_columns_added() -> None:
    """Kept rows carry a real timestamp plus the derived helper columns."""
    with tempfile.TemporaryDirectory() as tmp:
        frame, _, _ = cd.clean(cd.load_raw(write_raw(Path(tmp))))

    assert pd.api.types.is_datetime64_any_dtype(frame["published_at"])
    assert frame["published_at"].notna().all()

    gemini = frame[frame["url"].str.endswith("gemini")].iloc[0]
    assert gemini["published_year"] == 2026
    assert gemini["published_month"] == "2026-08"
    assert gemini["tag_count"] == 2

    untagged = frame[frame["url"].str.endswith("untagged")].iloc[0]
    assert untagged["tag_count"] == 0


def test_clean_output_is_sorted_newest_first() -> None:
    """Rows come out ordered by publication date, newest first."""
    with tempfile.TemporaryDirectory() as tmp:
        frame, _, _ = cd.clean(cd.load_raw(write_raw(Path(tmp))))

    dates = list(frame["published_at"])
    assert dates == sorted(dates, reverse=True)


# --------------------------------------------------------------------------- #
# Tags
# --------------------------------------------------------------------------- #


def test_split_tags() -> None:
    """Pipe-separated tags become clean lists; empty input becomes ``[]``."""
    series = pd.Series(["A|B", "", "  ", "A | B ", "Solo", "A||B"])
    result = cd.split_tags(series)

    assert list(result) == [["A", "B"], [], [], ["A", "B"], ["Solo"], ["A", "B"]]
    assert result.apply(len).tolist() == [2, 0, 0, 2, 1, 2]


def test_tag_frequencies() -> None:
    """Tag counts are per article and sorted from most to least frequent."""
    frame = pd.DataFrame({"tags": ["A|B", "A", "", "B|C|A"]})
    counts = cd.tag_frequencies(frame)

    assert counts.loc["A"] == 3
    assert counts.loc["B"] == 2
    assert counts.loc["C"] == 1
    assert list(counts.index)[0] == "A"


def test_tag_frequencies_on_untagged_corpus() -> None:
    """A corpus without any tags returns an empty Series, not an error."""
    assert cd.tag_frequencies(pd.DataFrame({"tags": ["", ""]})).empty


# --------------------------------------------------------------------------- #
# Validation and round trip
# --------------------------------------------------------------------------- #


def test_validate_clean_catches_duplicates() -> None:
    """The final assertion layer rejects a frame with repeated URLs."""
    with tempfile.TemporaryDirectory() as tmp:
        frame, _, _ = cd.clean(cd.load_raw(write_raw(Path(tmp))))

    cd.validate_clean(frame)

    broken = pd.concat([frame, frame.head(1)], ignore_index=True)
    try:
        cd.validate_clean(broken)
    except AssertionError as exc:
        assert "Duplicate URLs" in str(exc)
    else:
        raise AssertionError("Expected AssertionError for duplicate URLs")


def test_main_is_idempotent() -> None:
    """Running the script twice produces byte-identical output."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        raw = write_raw(directory)
        out = directory / "processed" / "articles_clean.csv"
        argv = [
            "--input", str(raw),
            "--output", str(out),
            "--dropped-output", str(directory / "processed" / "articles_dropped.csv"),
            "--report-output", str(directory / "processed" / "cleaning_report.json"),
        ]

        assert cd.main(argv) == 0
        first = out.read_bytes()
        assert cd.main(argv) == 0
        assert out.read_bytes() == first

        # tags stay pipe-separated on disk and split back into lists on load.
        text = out.read_text(encoding="utf-8")
        assert "Artificial Intelligence|Large Language Models" in text

        loaded = cd.load_clean(out)
        assert len(loaded) == 3
        assert loaded.loc[loaded["url"].str.endswith("gemini"), "tags_list"].iloc[0] == [
            "Artificial Intelligence",
            "Large Language Models",
        ]


def test_main_reports_missing_input() -> None:
    """A missing raw file exits with code 2 instead of raising."""
    with tempfile.TemporaryDirectory() as tmp:
        assert cd.main(["--input", str(Path(tmp) / "nope.csv")]) == 2


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def main() -> int:
    """Run every ``test_`` function in this module.

    Returns:
        ``0`` if all tests pass, ``1`` otherwise.
    """
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for name, test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
