#!/usr/bin/env python3
"""Turn ``data/raw/articles_raw.csv`` into an NLP-ready dataset.

The raw file is written by ``scraping/fetch_articles.py`` and mixes real blog
posts with a handful of DataCamp *certification landing pages*
(``sql-certification``, ``power-bi-certification``, ``GitHub-certifications``).
Those pages live under ``/blog/`` and therefore pass the scraper's URL filter,
but they carry no ``datePublished``, no author and no taxonomy -- so an empty
``published_date`` is the reliable marker that separates them from articles.

Cleaning is a chain of small, independently reportable filters:

    missing url -> missing date -> unparsable date -> empty title
                -> empty excerpt -> duplicate url

Every dropped row is kept with the reason it was dropped, so the removal count
is auditable instead of being a silent difference between two row counts.

The CSV is parsed with :func:`pandas.read_csv` (a proper RFC 4180 reader) --
titles such as ``"Gemini 3.7 Flash: Features, Benchmarks, Pricing"`` contain
commas inside quotes and would break a naive ``line.split(",")``.

Usage:
    python analysis/clean_data.py
    python analysis/clean_data.py --input data/raw/articles_raw.csv -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_INPUT = Path("data/raw/articles_raw.csv")
DEFAULT_OUTPUT = Path("data/processed/articles_clean.csv")
DEFAULT_DROPPED_OUTPUT = Path("data/processed/articles_dropped.csv")
DEFAULT_REPORT_OUTPUT = Path("data/processed/cleaning_report.json")

#: Separator used by ``fetch_articles.py`` when it flattens the tag list.
TAG_SEPARATOR = "|"

#: Date format emitted by ``fetch_articles.parse_date``.
DATE_FORMAT = "%Y-%m-%d"

#: Columns the raw file must provide.
REQUIRED_COLUMNS = ("title", "url", "published_date", "author", "tags", "excerpt", "scraped_at")

#: Extra columns added by the cleaning step.
DERIVED_COLUMNS = ("published_year", "published_month", "tag_count")

#: Name of the column that records why a row was removed.
DROP_REASON_COLUMN = "drop_reason"

LOGGER = logging.getLogger("clean_data")


class SchemaError(ValueError):
    """Raised when the raw CSV does not have the expected columns."""


@dataclass
class CleaningReport:
    """Row counts and drop reasons collected while cleaning.

    Attributes:
        rows_in: Number of rows read from the raw file.
        rows_out: Number of rows kept after every filter.
        dropped: Number of rows removed per reason, in filter order.
        examples: Up to a few example URLs per reason, for eyeballing.
    """

    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def rows_dropped(self) -> int:
        """Total number of removed rows.

        Returns:
            Sum of all per-reason drop counts.
        """
        return sum(self.dropped.values())

    def record(self, reason: str, removed: pd.DataFrame) -> None:
        """Register the rows removed by one filter.

        Args:
            reason: Human-readable filter name.
            removed: The rows that the filter rejected.
        """
        if removed.empty:
            return
        self.dropped[reason] = self.dropped.get(reason, 0) + len(removed)
        urls = [str(url) for url in removed.get("url", pd.Series(dtype=str)).head(3)]
        self.examples.setdefault(reason, []).extend(urls)

    def to_dict(self) -> dict[str, object]:
        """Serialize the report.

        Returns:
            A JSON-friendly dictionary.
        """
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_dropped,
            "dropped_by_reason": self.dropped,
            "examples": self.examples,
        }

    def format_text(self) -> str:
        """Render the report as a short human-readable block.

        Returns:
            Multi-line summary text.
        """
        lines = [
            f"rows in : {self.rows_in}",
            f"rows out: {self.rows_out}",
            f"removed : {self.rows_dropped}",
        ]
        for reason, count in self.dropped.items():
            examples = ", ".join(self.examples.get(reason, [])[:3])
            lines.append(f"  - {reason}: {count}" + (f"  (e.g. {examples})" if examples else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV without letting pandas invent missing values.

    ``keep_default_na=False`` matters: the scraper writes ``""`` for fields it
    could not find, and the default behaviour would turn those into ``NaN``,
    which then silently propagates into every downstream string operation.

    Args:
        path: Path to ``articles_raw.csv``.

    Returns:
        A DataFrame of stripped strings with the raw schema.

    Raises:
        FileNotFoundError: If the file does not exist.
        SchemaError: If any required column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scraping/fetch_articles.py first")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"{path} is missing columns: {', '.join(missing)}")

    frame = frame[list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()

    LOGGER.info("Read %d rows from %s", len(frame), path)
    return frame


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def drop_blank(frame: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame on whether one column is empty.

    Args:
        frame: Input rows.
        column: Column to test.

    Returns:
        A ``(kept, removed)`` pair.
    """
    blank = frame[column].str.len() == 0
    return frame[~blank].copy(), frame[blank].copy()


def parse_published_dates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse ``published_date`` and reject anything that is not a real date.

    The scraper already normalizes dates to ``YYYY-MM-DD``, so parsing is
    strict on purpose: an unexpected format means the scraper changed and
    should be noticed, not quietly reinterpreted.

    Args:
        frame: Rows with a non-empty ``published_date``.

    Returns:
        A ``(kept, removed)`` pair; ``kept`` gains a datetime ``published_at``
        column alongside the original string.
    """
    parsed = pd.to_datetime(frame["published_date"], format=DATE_FORMAT, errors="coerce")
    invalid = parsed.isna()

    if invalid.any():
        LOGGER.warning(
            "Unparsable published_date values (expected %s): %s",
            DATE_FORMAT,
            sorted(frame.loc[invalid, "published_date"].unique())[:5],
        )

    kept = frame[~invalid].copy()
    kept["published_at"] = parsed[~invalid]
    return kept, frame[invalid].copy()


def drop_duplicate_urls(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the first row per URL.

    ``fetch_articles.merge_rows`` already deduplicates at collection time; this
    is a second line of defence in case the raw file was hand-edited or two
    scrape runs were concatenated.

    Args:
        frame: Input rows.

    Returns:
        A ``(kept, removed)`` pair.
    """
    duplicated = frame.duplicated(subset="url", keep="first")
    if duplicated.any():
        LOGGER.warning("Found %d duplicate URLs in the raw file", int(duplicated.sum()))
    return frame[~duplicated].copy(), frame[duplicated].copy()


# --------------------------------------------------------------------------- #
# Tags
# --------------------------------------------------------------------------- #


def split_tags(tags: pd.Series) -> pd.Series:
    """Turn the pipe-separated ``tags`` column into lists of tags.

    An untagged article yields ``[]`` rather than ``[""]``, so
    ``Series.explode`` and ``len`` both behave.

    Args:
        tags: The ``tags`` column as strings.

    Returns:
        A Series of ``list[str]`` aligned with the input index.
    """
    return (
        tags.fillna("")
        .astype(str)
        .str.split(TAG_SEPARATOR)
        .apply(lambda parts: [part.strip() for part in parts if part.strip()])
    )


def tag_frequencies(frame: pd.DataFrame) -> pd.Series:
    """Count how many articles carry each tag.

    Args:
        frame: A cleaned frame containing a ``tags`` column.

    Returns:
        Tag counts sorted from most to least frequent.
    """
    exploded = split_tags(frame["tags"]).explode().dropna()
    if exploded.empty:
        return pd.Series(dtype="int64")
    return exploded.value_counts()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the convenience columns used by the analysis steps.

    Args:
        frame: Cleaned rows carrying a datetime ``published_at`` column.

    Returns:
        The frame with ``published_year``, ``published_month`` and
        ``tag_count`` appended.
    """
    frame = frame.copy()
    frame["published_year"] = frame["published_at"].dt.year.astype("int64")
    frame["published_month"] = frame["published_at"].dt.strftime("%Y-%m")
    frame["tag_count"] = split_tags(frame["tags"]).apply(len).astype("int64")
    return frame


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Run every filter and collect an audit trail.

    Args:
        frame: Raw rows as returned by :func:`load_raw`.

    Returns:
        A ``(clean_frame, dropped_frame, report)`` triple.  ``dropped_frame``
        carries a :data:`DROP_REASON_COLUMN` column.
    """
    report = CleaningReport(rows_in=len(frame))
    dropped: list[pd.DataFrame] = []

    def apply(reason: str, kept: pd.DataFrame, removed: pd.DataFrame) -> pd.DataFrame:
        report.record(reason, removed)
        if not removed.empty:
            tagged = removed.copy()
            tagged[DROP_REASON_COLUMN] = reason
            dropped.append(tagged)
        return kept

    frame = apply("missing url", *drop_blank(frame, "url"))
    # Certification landing pages are exactly the rows without a publish date.
    frame = apply("missing published_date (certification page)", *drop_blank(frame, "published_date"))
    frame = apply("unparsable published_date", *parse_published_dates(frame))
    frame = apply("empty title", *drop_blank(frame, "title"))
    frame = apply("empty excerpt", *drop_blank(frame, "excerpt"))
    frame = apply("duplicate url", *drop_duplicate_urls(frame))

    frame = add_derived_columns(frame)
    frame = frame.sort_values(["published_at", "url"], ascending=[False, True]).reset_index(drop=True)

    report.rows_out = len(frame)

    columns = list(REQUIRED_COLUMNS) + [DROP_REASON_COLUMN]
    dropped_frame = (
        pd.concat(dropped, ignore_index=True)[columns]
        if dropped
        else pd.DataFrame(columns=columns)
    )
    return frame, dropped_frame, report


def validate_clean(frame: pd.DataFrame) -> None:
    """Assert the invariants the analysis steps rely on.

    Args:
        frame: The cleaned frame.

    Raises:
        AssertionError: If any invariant is violated.
    """
    assert not frame.empty, "Cleaning removed every row - check the raw file"

    expected = set(REQUIRED_COLUMNS) | set(DERIVED_COLUMNS)
    assert expected <= set(frame.columns), f"Missing columns: {sorted(expected - set(frame.columns))}"

    assert frame["url"].is_unique, "Duplicate URLs survived cleaning"
    for column in ("title", "url", "excerpt", "published_date"):
        assert (frame[column].str.len() > 0).all(), f"Empty {column} survived cleaning"

    reparsed = pd.to_datetime(frame["published_date"], format=DATE_FORMAT, errors="coerce")
    assert reparsed.notna().all(), "Unparsable published_date survived cleaning"
    assert (frame["tag_count"] >= 0).all(), "Negative tag_count"


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame to CSV, creating parent directories as needed.

    ``tags`` stays pipe-separated on disk: a CSV cell holding ``"['a', 'b']"``
    would have to be parsed back with ``ast.literal_eval``, which is both
    fragile and lossy.  Call :func:`split_tags` after loading instead.

    Args:
        frame: Rows to write.
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [column for column in frame.columns if column != "published_at"]
    frame[columns].to_csv(path, index=False, encoding="utf-8")
    LOGGER.info("Wrote %d rows to %s", len(frame), path)


def write_report(report: CleaningReport, path: Path) -> None:
    """Persist the cleaning report as JSON.

    Args:
        report: The collected report.
        path: Destination JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote cleaning report to %s", path)


def load_clean(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load the cleaned dataset with tags already split into lists.

    Convenience entry point for notebooks and the downstream analysis modules.

    Args:
        path: Path to ``articles_clean.csv``.

    Returns:
        The cleaned frame with an extra ``tags_list`` column of ``list[str]``
        and a datetime ``published_at`` column.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    frame["tags_list"] = split_tags(frame["tags"])
    frame["published_at"] = pd.to_datetime(frame["published_date"], format=DATE_FORMAT)
    frame["tag_count"] = frame["tags_list"].apply(len).astype("int64")
    frame["published_year"] = frame["published_at"].dt.year.astype("int64")
    return frame


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="raw CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="cleaned CSV path")
    parser.add_argument(
        "--dropped-output",
        type=Path,
        default=DEFAULT_DROPPED_OUTPUT,
        help="where to write the removed rows with their drop reason",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
        help="where to write the JSON cleaning report",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        raw = load_raw(args.input)
    except (FileNotFoundError, SchemaError) as exc:
        LOGGER.error("%s", exc)
        return 2

    frame, dropped, report = clean(raw)
    validate_clean(frame)

    write_csv(frame, args.output)
    write_csv(dropped, args.dropped_output)
    write_report(report, args.report_output)

    print(report.format_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
