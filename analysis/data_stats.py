#!/usr/bin/env python3
"""Precheck report for the cleaned article dataset.

Run this straight after ``clean_data.py`` and before any topic modelling: it
answers the questions that decide whether LDA is even worth running -- how much
data survived cleaning, whether the corpus is dominated by one period, and how
long the tag tail is.

Reported sections:

    1. What cleaning removed and why (from ``cleaning_report.json``)
    2. Corpus size, date range and text-length distribution
    3. Publication dates per year, with a bar sparkline
    4. Top-20 tags, tag coverage and tag-count distribution

Usage:
    python analysis/data_stats.py
    python analysis/data_stats.py --top 30 --output data/processed/data_stats.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

# Keep the module importable both as `python analysis/data_stats.py` and from a
# notebook whose working directory is the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_data import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_REPORT_OUTPUT,
    load_clean,
    tag_frequencies,
)

DEFAULT_TOP_TAGS = 20

#: Width of the ASCII bars used for the yearly histogram.
BAR_WIDTH = 40


def read_report(path: Path) -> dict[str, object] | None:
    """Load the JSON cleaning report if it is available.

    Args:
        path: Path to ``cleaning_report.json``.

    Returns:
        The decoded report, or ``None`` when the file is absent or malformed.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def bar(value: int, maximum: int, width: int = BAR_WIDTH) -> str:
    """Render a proportional ASCII bar.

    Args:
        value: Value to draw.
        maximum: Value that corresponds to a full-width bar.
        width: Full-width length in characters.

    Returns:
        A string of block characters, at least one wide for a non-zero value.
    """
    if maximum <= 0 or value <= 0:
        return ""
    return "█" * max(1, round(value / maximum * width))


def section(title: str) -> list[str]:
    """Build a section header.

    Args:
        title: Header text.

    Returns:
        The header lines.
    """
    return ["", title, "-" * len(title)]


def format_cleaning_summary(report: dict[str, object] | None) -> list[str]:
    """Describe how many rows cleaning removed, and why.

    Args:
        report: Decoded cleaning report, or ``None``.

    Returns:
        Report lines.
    """
    lines = section("1. Cleaning summary")
    if report is None:
        lines.append(f"(no report found at {DEFAULT_REPORT_OUTPUT} - run clean_data.py first)")
        return lines

    rows_in = int(report.get("rows_in", 0))
    rows_out = int(report.get("rows_out", 0))
    removed = int(report.get("rows_dropped", 0))
    share = removed / rows_in * 100 if rows_in else 0.0

    lines.append(f"raw rows     : {rows_in}")
    lines.append(f"clean rows   : {rows_out}")
    lines.append(f"removed      : {removed} ({share:.1f}%)")

    by_reason = report.get("dropped_by_reason") or {}
    examples = report.get("examples") or {}
    if isinstance(by_reason, dict) and by_reason:
        lines.append("removed by reason:")
        for reason, count in by_reason.items():
            lines.append(f"  {count:>5}  {reason}")
            for url in (examples.get(reason) or [])[:3]:
                lines.append(f"         e.g. {url}")
    else:
        lines.append("removed by reason: nothing was dropped")
    return lines


def format_corpus_overview(frame: pd.DataFrame) -> list[str]:
    """Summarize corpus size and the text fields LDA will consume.

    Args:
        frame: Cleaned frame from :func:`clean_data.load_clean`.

    Returns:
        Report lines.
    """
    lines = section("2. Corpus overview")
    lines.append(f"articles     : {len(frame)}")
    lines.append(f"unique URLs  : {frame['url'].nunique()}")
    lines.append(f"unique authors: {frame['author'].replace('', pd.NA).nunique()}")
    lines.append(
        f"date range   : {frame['published_at'].min():%Y-%m-%d}"
        f" .. {frame['published_at'].max():%Y-%m-%d}"
    )

    for column in ("title", "excerpt"):
        words = frame[column].str.split().apply(len)
        lines.append(
            f"{column:<9} words: mean {words.mean():.1f}, median {words.median():.0f}, "
            f"min {words.min()}, max {words.max()}"
        )

    missing_author = int((frame["author"].str.len() == 0).sum())
    if missing_author:
        lines.append(f"note         : {missing_author} articles have no author")
    return lines


def format_date_distribution(frame: pd.DataFrame) -> list[str]:
    """Show how the articles are spread across publication years.

    A corpus skewed towards the last two years changes how topic drift should
    be read, so this is checked before modelling rather than after.

    Args:
        frame: Cleaned frame.

    Returns:
        Report lines.
    """
    lines = section("3. Publication dates")
    per_year = frame["published_year"].value_counts().sort_index()
    peak = int(per_year.max()) if not per_year.empty else 0

    for year, count in per_year.items():
        share = count / len(frame) * 100
        lines.append(f"{int(year)}  {count:>5}  {share:>5.1f}%  {bar(int(count), peak)}")

    last_year = frame["published_at"].max() - pd.DateOffset(years=1)
    recent = int((frame["published_at"] > last_year).sum())
    lines.append(f"published in the last 12 months: {recent} ({recent / len(frame) * 100:.1f}%)")
    return lines


def format_tag_stats(frame: pd.DataFrame, top: int = DEFAULT_TOP_TAGS) -> list[str]:
    """Report tag coverage and the most frequent tags.

    Args:
        frame: Cleaned frame.
        top: How many tags to list.

    Returns:
        Report lines.
    """
    lines = section(f"4. Tags (top {top})")

    counts = tag_frequencies(frame)
    untagged = int((frame["tag_count"] == 0).sum())

    lines.append(f"distinct tags: {len(counts)}")
    lines.append(f"untagged articles: {untagged} ({untagged / len(frame) * 100:.1f}%)")
    lines.append(f"tags per article: mean {frame['tag_count'].mean():.2f}, max {frame['tag_count'].max()}")

    if counts.empty:
        lines.append("(no tags in the dataset)")
        return lines

    singletons = int((counts == 1).sum())
    lines.append(f"tags used by a single article: {singletons} ({singletons / len(counts) * 100:.1f}%)")
    lines.append("")

    peak = int(counts.iloc[0])
    for rank, (tag, count) in enumerate(counts.head(top).items(), start=1):
        share = count / len(frame) * 100
        lines.append(f"{rank:>3}. {str(tag):<28} {count:>5}  {share:>5.1f}%  {bar(int(count), peak)}")
    return lines


def build_report(frame: pd.DataFrame, report: dict[str, object] | None, top: int) -> str:
    """Assemble the full text report.

    Args:
        frame: Cleaned frame.
        report: Decoded cleaning report, or ``None``.
        top: How many tags to list.

    Returns:
        The report as one string.
    """
    lines = ["DataCamp blog - cleaned dataset precheck", "=" * 42]
    lines += format_cleaning_summary(report)
    lines += format_corpus_overview(frame)
    lines += format_date_distribution(frame)
    lines += format_tag_stats(frame, top)
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT, help="cleaned CSV path")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
        help="JSON cleaning report written by clean_data.py",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_TAGS, help="how many tags to list")
    parser.add_argument("--output", type=Path, default=None, help="also write the report to this file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"{args.input} not found - run analysis/clean_data.py first", file=sys.stderr)
        return 2

    frame = load_clean(args.input)
    if frame.empty:
        print(f"{args.input} has no rows", file=sys.stderr)
        return 1

    text = build_report(frame, read_report(args.report), args.top)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Saved to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
