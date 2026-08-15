#!/usr/bin/env python3
"""Offline checks for scraping/fetch_articles.py.

Every test runs against an HTML fixture -- no network access is required.

Run with:
    python tests/test_fetch_articles.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scraping"))

import fetch_articles as fa  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

#: The site-wide category menu that used to leak into every row.  Both spellings
#: of the "See More" control are present: a plain one and one whose icon lives in
#: a separate span (the source of the fused "See MoreRight Arrow" value).
SITE_WIDE_CATEGORY_MENU = (
    '<div class="blog-categories-list">'
    + "".join(
        f'<a href="/blog/category/menu-topic-{i}">Menu Topic {i}</a>' for i in range(70)
    )
    + '<a href="/blog/category/all">See More</a>'
    + '<a href="/blog/category/all2"><span>See More</span><span>Right Arrow</span></a>'
    + "</div>"
)

GEMINI_PAGE_PROPS = {
    "props": {
        "pageProps": {
            "page": {
                "title": "Gemini 3.7 Flash: Tests, Features and Access",
                "slug": "gemini-3-7-flash",
                "category": {"id": 16, "tag": "Artificial Intelligence", "slug": "ai"},
                "subCategories": [
                    {"id": 204, "tag": "Large Language Models", "slug": "large-language-models"}
                ],
            }
        }
    }
}

#: JSON-LD that dumps the entire site taxonomy into `keywords`.  Level 1 must be
#: rejected on size so it cannot outrank anything below it.
BLOATED_JSONLD = json.dumps(
    {
        "@type": "BlogPosting",
        "headline": "Gemini 3.7 Flash: Tests, Features and Access",
        "datePublished": "2026-02-11T09:00:00Z",
        "author": {"@type": "Person", "name": "Alex Olteanu"},
        "description": "Hands-on tests of Gemini 3.7 Flash.",
        "keywords": ", ".join(f"Menu Topic {i}" for i in range(70)),
    }
)


def next_data_script(payload: object) -> str:
    """Render a ``__NEXT_DATA__`` script tag.

    Args:
        payload: Value to serialize, or a raw string to embed verbatim.

    Returns:
        The script tag as HTML.
    """
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return f'<script id="__NEXT_DATA__" type="application/json">{body}</script>'


def article_page(
    *,
    next_data: str | None = None,
    jsonld: str | None = None,
    meta: str = "",
    body: str = "",
) -> str:
    """Build an article page fixture.

    Args:
        next_data: Raw ``__NEXT_DATA__`` script tag, if any.
        jsonld: Raw JSON-LD payload, if any.
        meta: Extra ``<meta>`` markup.
        body: Extra markup placed inside ``<article>``.

    Returns:
        A full HTML document.
    """
    head = meta
    if jsonld is not None:
        head += f'<script type="application/ld+json">{jsonld}</script>'
    if next_data is not None:
        head += next_data
    return (
        "<html><head><title>page</title>"
        f"{head}</head><body>"
        "<nav><a href='/blog/category/nav-noise'>Nav Noise</a></nav>"
        "<main><article><h1>Gemini 3.7 Flash: Tests, Features and Access</h1>"
        f"{body}"
        "<p>Gemini 3.7 Flash is a fast multimodal model that we put through a "
        "series of practical benchmarks.</p>"
        "</article></main>"
        f"{SITE_WIDE_CATEGORY_MENU}"
        "</body></html>"
    )


#: Real shape of a listing page: `props.pageProps.blogs` holds exactly the 12
#: articles of that page, `count` is the site-wide article total (1502 as of
#: 2026-08-15, i.e. 126 pages).  `mostRecentBlogs` is a *different* section that
#: the markup renders too -- it must not end up in the page's link list.
LISTING_PER_PAGE = 12
LISTING_COUNT = 1502
LISTING_TOTAL_PAGES = 126

LISTING_PAGE_PROPS = {
    "props": {
        "pageProps": {
            "count": LISTING_COUNT,
            "blogs": [
                {"slug": f"page-2-article-{i}", "title": f"Page 2 Article {i}"}
                for i in range(LISTING_PER_PAGE)
            ],
            "mostRecentBlogs": [
                {"slug": f"most-recent-{i}", "title": f"Most Recent {i}"} for i in range(9)
            ],
        }
    }
}


def listing_page(payload: object = LISTING_PAGE_PROPS) -> str:
    """Build a listing page fixture.

    The markup renders both the paginated section and the "most recent" strip,
    which is why link extraction must not trust it.

    Args:
        payload: Value embedded in the ``__NEXT_DATA__`` script tag.

    Returns:
        A full HTML document.
    """
    props = payload.get("props", {}).get("pageProps", {}) if isinstance(payload, dict) else {}
    anchors = "".join(
        f'<a href="/blog/{entry["slug"]}">{entry["title"]}</a>'
        for key in ("blogs", "mostRecentBlogs")
        for entry in props.get(key, [])
    )
    return (
        "<html><head>"
        f"{next_data_script(payload)}"
        "</head><body><main>"
        f"{anchors}"
        '<a href="/blog/page/3">Next</a>'
        "</main>"
        f"{SITE_WIDE_CATEGORY_MENU}"
        "</body></html>"
    )


ARTICLE_TAG_META = (
    '<meta property="article:tag" content="Artificial Intelligence">'
    '<meta property="article:tag" content="Large Language Models">'
)

EXPECTED_TAGS = ["Artificial Intelligence", "Large Language Models"]


# --------------------------------------------------------------------------- #
# Level 0: __NEXT_DATA__
# --------------------------------------------------------------------------- #


def test_next_data_tags_from_category_and_subcategories() -> None:
    """category.tag plus every subCategories[].tag, in that order."""
    assert fa.tags_from_next_data(GEMINI_PAGE_PROPS) == EXPECTED_TAGS


def test_next_data_wins_over_bloated_jsonld_and_site_menu() -> None:
    """Level 0 decides the tags even when the page is full of noise."""
    html = article_page(
        next_data=next_data_script(GEMINI_PAGE_PROPS),
        jsonld=BLOATED_JSONLD,
        meta=ARTICLE_TAG_META,
    )
    row = fa.parse_article_page(html, "https://www.datacamp.com/blog/gemini-3-7-flash")
    assert row["tags"] == "Artificial Intelligence|Large Language Models", row["tags"]
    assert "Menu Topic" not in row["tags"]
    assert "See More" not in row["tags"]


def test_next_data_category_only() -> None:
    """An article without subcategories yields a single tag."""
    payload = {"props": {"pageProps": {"page": {"category": {"tag": "Data Engineering"}}}}}
    assert fa.tags_from_next_data(payload) == ["Data Engineering"]


def test_next_data_missing_falls_back_to_meta() -> None:
    """Without __NEXT_DATA__ the meta level takes over."""
    html = article_page(jsonld=BLOATED_JSONLD, meta=ARTICLE_TAG_META)
    row = fa.parse_article_page(html, "https://www.datacamp.com/blog/x")
    assert row["tags"].split("|") == EXPECTED_TAGS


def test_next_data_malformed_json_does_not_crash() -> None:
    """Broken JSON is ignored, not fatal."""
    html = article_page(next_data=next_data_script("{not valid json"), meta=ARTICLE_TAG_META)
    soup = fa.make_soup(html)
    assert fa.find_next_data(soup) is None
    row = fa.parse_article_page(html, "https://www.datacamp.com/blog/x")
    assert row["tags"].split("|") == EXPECTED_TAGS


def test_next_data_without_page_path_falls_back() -> None:
    """A payload that lacks props.pageProps.page hands over to the next level."""
    payload = {"props": {"pageProps": {"other": {"category": {"tag": "Nope"}}}}}
    html = article_page(next_data=next_data_script(payload), meta=ARTICLE_TAG_META)
    assert fa.tags_from_next_data(payload) == []
    row = fa.parse_article_page(html, "https://www.datacamp.com/blog/x")
    assert row["tags"].split("|") == EXPECTED_TAGS


def test_oversized_next_data_taxonomy_is_rejected() -> None:
    """Even level 0 is size-checked, so a leaked taxonomy cannot win."""
    payload = {
        "props": {
            "pageProps": {
                "page": {
                    "category": {"tag": "Artificial Intelligence"},
                    "subCategories": [{"tag": f"Menu Topic {i}"} for i in range(70)],
                }
            }
        }
    }
    html = article_page(next_data=next_data_script(payload), meta=ARTICLE_TAG_META)
    row = fa.parse_article_page(html, "https://www.datacamp.com/blog/x")
    assert row["tags"].split("|") == EXPECTED_TAGS


def test_article_links_recovered_from_next_data_walk() -> None:
    """A client-rendered listing without `blogs` still yields links."""
    payload = {
        "props": {
            "pageProps": {
                "articles": [
                    {"slug": "gemini-3-7-flash", "title": "Gemini 3.7 Flash"},
                    {"slug": "what-is-rag", "title": "What is RAG?"},
                    {"slug": "no-title-here"},
                ]
            }
        }
    }
    html = f"<html><body>{next_data_script(payload)}</body></html>"
    links = fa.extract_article_links(html, "https://www.datacamp.com/blog")
    assert links == [
        "https://www.datacamp.com/blog/gemini-3-7-flash",
        "https://www.datacamp.com/blog/what-is-rag",
    ], links


# --------------------------------------------------------------------------- #
# Listing pages: pagination and link source
# --------------------------------------------------------------------------- #


def test_listing_url_uses_path_pagination() -> None:
    """/blog for page 1, /blog/page/N after that - never the ignored ?page=."""
    assert fa.listing_url(1) == "https://www.datacamp.com/blog"
    assert fa.listing_url(2) == "https://www.datacamp.com/blog/page/2"
    assert fa.listing_url(126) == "https://www.datacamp.com/blog/page/126"
    assert "?page=" not in fa.listing_url(3)


def test_listing_page_url_is_not_an_article() -> None:
    """The pagination URL must not be mistaken for an article."""
    assert not fa.is_article_url(fa.listing_url(2))


def test_blogs_is_the_link_source_not_the_markup() -> None:
    """Exactly the 12 paginated articles, without the "most recent" strip."""
    page_url = fa.listing_url(2)
    html = listing_page()

    links = fa.extract_article_links(html, page_url)
    assert len(links) == LISTING_PER_PAGE, len(links)
    assert links == [
        f"https://www.datacamp.com/blog/page-2-article-{i}" for i in range(LISTING_PER_PAGE)
    ], links
    assert not any("most-recent" in link for link in links)

    # The markup mixes both sections - this is the 21-vs-12 regression.
    from_markup = fa.dedupe(fa.links_from_markup(fa.make_soup(html), page_url))
    assert len(from_markup) == LISTING_PER_PAGE + 9, len(from_markup)


def test_listing_stats_reads_count_and_per_page() -> None:
    """count and len(blogs) come straight from props.pageProps."""
    assert fa.listing_stats(LISTING_PAGE_PROPS) == (LISTING_COUNT, LISTING_PER_PAGE)
    assert fa.listing_stats(GEMINI_PAGE_PROPS) == (None, None)
    assert fa.listing_stats(None) == (None, None)


def test_total_pages_from_real_numbers() -> None:
    """1502 articles / 12 per page = 126 pages (the last one partial)."""
    assert fa.total_listing_pages(LISTING_COUNT, LISTING_PER_PAGE) == LISTING_TOTAL_PAGES
    assert fa.total_listing_pages(24, 12) == 2
    assert fa.total_listing_pages(25, 12) == 3
    assert fa.total_listing_pages(None, 12) is None
    assert fa.total_listing_pages(1502, None) is None
    assert fa.total_listing_pages(1502, 0) is None


def test_missing_blogs_falls_back_to_markup() -> None:
    """No props.pageProps.blogs - the markup takes over instead of failing."""
    payload = {"props": {"pageProps": {"count": LISTING_COUNT}}}
    html = (
        f"<html><head>{next_data_script(payload)}</head><body>"
        '<a href="/blog/what-is-rag">RAG</a>'
        '<a href="/blog/category/ai">AI</a>'
        "</body></html>"
    )
    assert fa.extract_article_links(html, fa.listing_url(1)) == [
        "https://www.datacamp.com/blog/what-is-rag"
    ]
    assert fa.listing_stats(payload) == (LISTING_COUNT, None)
    assert fa.total_listing_pages(*fa.listing_stats(payload)) is None


def test_blogs_entries_without_slug_are_skipped() -> None:
    """Malformed entries drop out instead of producing broken URLs."""
    payload = {
        "props": {
            "pageProps": {
                "blogs": [
                    {"slug": "what-is-rag", "title": "RAG"},
                    {"title": "No slug"},
                    {"slug": "   "},
                    "not-an-object",
                    {"slug": "category/ai", "title": "A listing, not an article"},
                ]
            }
        }
    }
    assert fa.article_links_from_blogs(payload) == ["https://www.datacamp.com/blog/what-is-rag"]


# --------------------------------------------------------------------------- #
# Fallback levels: article tags vs site-wide navigation
# --------------------------------------------------------------------------- #


def test_jsonld_blogposting_is_an_article_type() -> None:
    """BlogPosting has no "article" substring, but level 1 must still see it."""
    assert fa.is_article_type("BlogPosting")
    assert fa.is_article_type("https://schema.org/BlogPosting")
    assert fa.is_article_type(["WebPage", "NewsArticle"])
    assert not fa.is_article_type("BreadcrumbList")
    assert fa.jsonld_article(fa.make_soup(article_page(jsonld=BLOATED_JSONLD))) is not None


def test_jsonld_taxonomy_is_rejected_on_size_not_on_type() -> None:
    """The bloated keywords list reaches level 1 and loses the size check."""
    html = article_page(jsonld=BLOATED_JSONLD, meta=ARTICLE_TAG_META)
    soup = fa.make_soup(html)
    assert len(fa.tags_from_jsonld(soup)) > fa.MAX_TAGS
    row = fa.parse_article_page(html, "https://www.datacamp.com/blog/x")
    assert row["tags"].split("|") == EXPECTED_TAGS


def test_html_tag_block_beats_site_menu() -> None:
    """A declared tag block inside <article> wins over the global menu."""
    html = article_page(
        body='<div class="article-tags">'
        '<a href="/blog/category/ai">Artificial Intelligence</a>'
        '<a href="/blog/category/large-language-models">Large Language Models</a>'
        "</div>"
    )
    assert fa.tags_from_html(fa.make_soup(html)) == EXPECTED_TAGS


def test_site_menu_inside_main_rejected_on_size() -> None:
    """Without <article> the menu shares the scope and is dropped on size."""
    html = (
        "<html><body><main>"
        "<h1>Gemini 3.7 Flash</h1>"
        '<div class="post-meta">'
        '<a href="/blog/category/ai">Artificial Intelligence</a>'
        '<a href="/blog/category/large-language-models">Large Language Models</a>'
        "</div>"
        f"{SITE_WIDE_CATEGORY_MENU}"
        "</main></body></html>"
    )
    assert fa.tags_from_html(fa.make_soup(html)) == EXPECTED_TAGS


def test_only_site_menu_yields_no_tags() -> None:
    """When the article has no tags the column stays empty, never the menu."""
    html = article_page()
    row = fa.parse_article_page(html, "https://www.datacamp.com/blog/x")
    assert row["tags"] == "", row["tags"]


def test_icon_spans_do_not_fuse_into_labels() -> None:
    """get_text(' ') keeps 'See More' and 'Right Arrow' apart, both filtered."""
    labels = fa.link_labels(fa.make_soup(SITE_WIDE_CATEGORY_MENU).find_all("a"))
    assert not any("SeeMore" in label or "MoreRight" in label for label in labels)
    assert "See More Right Arrow" not in labels
    assert "See More" not in labels


def test_limit_tags_rejects_rather_than_truncates() -> None:
    """An oversized list is discarded whole, so bad data never looks like success."""
    assert fa.limit_tags([f"Topic {i}" for i in range(70)], "test") == []
    assert fa.limit_tags(EXPECTED_TAGS, "test") == EXPECTED_TAGS


# --------------------------------------------------------------------------- #
# URLs, dates, other metadata
# --------------------------------------------------------------------------- #


def test_canonical_url_strips_query_and_trailing_slash() -> None:
    """The same article always canonicalizes to the same key."""
    base = "https://www.datacamp.com/blog"
    assert fa.canonical_url("/blog/what-is-rag/?utm=x#top", base) == (
        "https://www.datacamp.com/blog/what-is-rag"
    )
    assert fa.canonical_url("mailto:hi@example.com", base) is None


def test_is_article_url_rejects_listings() -> None:
    """Category, tag and author listings are not articles."""
    assert fa.is_article_url("https://www.datacamp.com/blog/what-is-rag")
    assert not fa.is_article_url("https://www.datacamp.com/blog/category/ai")
    assert not fa.is_article_url("https://www.datacamp.com/blog")
    assert not fa.is_article_url("https://example.com/blog/what-is-rag")


def test_extract_article_links_filters_non_articles() -> None:
    """Only article links survive, deduplicated."""
    html = (
        "<html><body>"
        '<a href="/blog/what-is-rag">RAG</a>'
        '<a href="/blog/what-is-rag/">RAG again</a>'
        '<a href="/blog/category/ai">AI</a>'
        '<a href="/blog/author/someone">Someone</a>'
        '<a href="/courses/intro">Course</a>'
        "</body></html>"
    )
    assert fa.extract_article_links(html, "https://www.datacamp.com/blog") == [
        "https://www.datacamp.com/blog/what-is-rag"
    ]


def test_parse_date_formats() -> None:
    """ISO, ordinal and day-first dates all normalize to YYYY-MM-DD."""
    assert fa.parse_date("2024-05-21T10:00:00Z") == "2024-05-21"
    assert fa.parse_date("2024-05-21T10:00:00+02:00") == "2024-05-21"
    assert fa.parse_date("May 21st, 2024") == "2024-05-21"
    assert fa.parse_date("21 May 2024") == "2024-05-21"
    assert fa.parse_date("") == ""
    assert fa.parse_date("not a date") == ""


def test_title_author_date_and_excerpt() -> None:
    """Core metadata comes through and the excerpt is not the full body."""
    html = article_page(next_data=next_data_script(GEMINI_PAGE_PROPS), jsonld=BLOATED_JSONLD)
    row = fa.parse_article_page(html, "https://www.datacamp.com/blog/gemini-3-7-flash")
    assert row["title"] == "Gemini 3.7 Flash: Tests, Features and Access"
    assert row["published_date"] == "2026-02-11"
    assert row["author"] == "Alex Olteanu"
    assert row["excerpt"] == "Hands-on tests of Gemini 3.7 Flash."
    assert set(row) == set(fa.COLUMNS)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def sample_row(url: str, title: str = "T", tags: str = "AI") -> dict[str, str]:
    """Build a minimal valid output row.

    Args:
        url: Article URL.
        title: Article title.
        tags: Pipe-separated tags.

    Returns:
        A row dictionary.
    """
    return {
        "title": title,
        "url": url,
        "published_date": "2024-05-21",
        "author": "A",
        "tags": tags,
        "excerpt": "E",
        "scraped_at": "2026-08-15",
    }


def test_merge_is_idempotent_and_prefers_fresh() -> None:
    """Re-scraping replaces a row instead of appending a duplicate."""
    old = [sample_row("https://www.datacamp.com/blog/a", title="Old")]
    new = [sample_row("https://www.datacamp.com/blog/a", title="New")]
    merged = fa.merge_rows(old, new)
    assert len(merged) == 1
    assert merged[0]["title"] == "New"


def test_csv_round_trip_is_stable() -> None:
    """Writing, reading back and merging leaves the file unchanged."""
    rows = [sample_row("https://www.datacamp.com/blog/a")]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "articles_raw.csv"
        fa.validate_rows(rows)
        fa.write_csv(rows, path)
        again = fa.merge_rows(fa.read_existing(path), rows)
        assert again == rows


def test_validate_rows_blocks_tag_regression() -> None:
    """A row carrying the whole menu must never reach the CSV."""
    bad = [sample_row("https://www.datacamp.com/blog/a", tags="|".join(f"T{i}" for i in range(70)))]
    try:
        fa.validate_rows(bad)
    except AssertionError:
        return
    raise AssertionError("validate_rows accepted an oversized tag list")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def main() -> int:
    """Run every test in this module.

    Returns:
        ``0`` if all tests passed, ``1`` otherwise.
    """
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
