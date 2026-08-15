#!/usr/bin/env python3
"""Collect public article metadata from the DataCamp blog.

The blog is a Next.js application: the page shell is server-rendered, but the
article taxonomy is hydrated on the client from a JSON blob embedded in
``<script id="__NEXT_DATA__" type="application/json">``.  Article tags live
*only* there -- they are absent from the rendered HTML, from ``<meta>`` tags and
from JSON-LD.  Any ``/category/`` or ``/tag/`` links found in the markup belong
to the site-wide navigation menu, not to the article.

Metadata extraction therefore runs through four ordered levels; the first
non-empty answer wins:

    0. ``__NEXT_DATA__`` -> ``props.pageProps.page``  (authoritative for tags)
    1. JSON-LD ``BlogPosting``/``Article``
    2. ``<meta>`` tags (OpenGraph, ``article:*``)
    3. HTML heuristics (``<h1>``, ``<time datetime>``, ``rel=author``, ...)

Every level is size-checked: a tag list longer than ``MAX_TAGS`` is discarded
outright (not truncated) and the next level gets its turn, so a page that leaks
the whole site taxonomy into ``keywords`` cannot poison the output.

The listing pages are hydrated the same way.  ``props.pageProps.blogs`` holds
exactly the articles of the requested page, while the markup also renders other
sections ("most recent", promoted posts), so the JSON is the authoritative link
source and the HTML is only a fallback.  Pagination uses the path form
``/blog/page/N`` (``/blog`` for page 1); the ``?page=`` query parameter is
ignored by the site and always returns page 1.

Only public metadata is collected -- article bodies are never downloaded.

Usage:
    python scraping/fetch_articles.py --pages 2 --verbose
    python scraping/fetch_articles.py --auto-pages --no-details
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

BASE_URL = "https://www.datacamp.com"
LISTING_PATH = "/blog"
#: Pagination is path-based (``/blog/page/2``).  ``/blog?page=2`` returns the
#: content of page 1 -- the query parameter is silently ignored by the site.
LISTING_PAGE_PATH = "/blog/page"
USER_AGENT = (
    "knowledgemap-datascience-ai/0.1 "
    "(+https://github.com/Data-Analytics-for-beginners/knowledgemap-datascience-ai; "
    "educational research; contact via GitHub issues)"
)
DEFAULT_OUTPUT = Path("data/raw/articles_raw.csv")

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
BACKOFF_SECONDS = 2
DEFAULT_DELAY_MIN = 1.0
DEFAULT_DELAY_MAX = 2.0
MIN_ALLOWED_DELAY = 1.0

#: Article-like JSON-LD ``@type`` names that do not contain the substring
#: "article".  ``BlogPosting`` is the type the DataCamp blog actually emits.
JSONLD_ARTICLE_TYPES = {"blogposting", "liveblogposting", "posting"}

#: A real article carries a handful of tags.  Anything above this is the site
#: navigation menu leaking in, so the whole list is rejected.
MAX_TAGS = 12
TAG_SEPARATOR = "|"

COLUMNS = ("title", "url", "published_date", "author", "tags", "excerpt", "scraped_at")

#: Listing paths that are not articles themselves.
NON_ARTICLE_SEGMENTS = {
    "category",
    "categories",
    "tag",
    "tags",
    "topic",
    "topics",
    "author",
    "authors",
    "page",
}

#: Boilerplate link labels that must never become tags.
TAG_STOPWORDS = {
    "see more",
    "see more right arrow",
    "see all",
    "view all",
    "view more",
    "show more",
    "browse",
    "browse all",
    "all",
    "more",
    "right arrow",
    "left arrow",
    "next",
    "previous",
}

LOGGER = logging.getLogger("fetch_articles")


class RobotsUnavailableError(RuntimeError):
    """Raised when ``robots.txt`` cannot be read, so crawling must not start."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def make_soup(html: str) -> BeautifulSoup:
    """Parse HTML with lxml, falling back to the stdlib parser.

    Args:
        html: Raw HTML document.

    Returns:
        A parsed BeautifulSoup tree.
    """
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # pragma: no cover - only when lxml is missing/broken
        return BeautifulSoup(html, "html.parser")


def dig(data: Any, *keys: str) -> Any:
    """Walk a nested mapping without raising on missing or wrongly-typed nodes.

    Args:
        data: Arbitrary decoded JSON value.
        *keys: Successive dictionary keys to follow.

    Returns:
        The value at the end of the path, or ``None`` if any step is missing.
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def clean_text(value: Any) -> str:
    """Collapse whitespace and return a stripped string.

    Args:
        value: Any value; non-strings become an empty string.

    Returns:
        Normalized text.
    """
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def dedupe(values: Iterable[str]) -> list[str]:
    """Drop case-insensitive duplicates while preserving order.

    Args:
        values: Strings to deduplicate.

    Returns:
        A list without repeated entries.
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def canonical_url(href: str, base: str) -> str | None:
    """Turn a possibly relative href into a canonical absolute URL.

    Query strings, fragments and trailing slashes are dropped so the same
    article always produces the same key when deduplicating.

    Args:
        href: Raw ``href`` attribute value.
        base: URL of the page the link was found on.

    Returns:
        Canonical URL, or ``None`` if the link is not http(s).
    """
    if not href:
        return None
    parts = urlsplit(urljoin(base, href.strip()))
    if parts.scheme not in ("http", "https"):
        return None
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme, parts.netloc.lower(), path, "", ""))


def is_article_url(url: str, base: str = BASE_URL) -> bool:
    """Check whether a canonical URL points at a single blog article.

    Args:
        url: Canonical absolute URL.
        base: Site root used to reject off-site links.

    Returns:
        ``True`` for ``/blog/<slug>``, ``False`` for listings and other hosts.
    """
    parts = urlsplit(url)
    if parts.netloc.lower() != urlsplit(base).netloc.lower():
        return False
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) != 2 or segments[0] != "blog":
        return False
    return segments[1].lower() not in NON_ARTICLE_SEGMENTS


def parse_date(raw: str) -> str:
    """Normalize a published date into ``YYYY-MM-DD``.

    Handles ISO 8601 (with ``Z`` or an offset), ``May 21st, 2024`` and
    ``21 May 2024``.

    Args:
        raw: Date string in an unknown format.

    Returns:
        ISO date string, or ``""`` if nothing could be parsed.
    """
    text = clean_text(raw)
    if not text:
        return ""

    iso_candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(iso_candidate).date().isoformat()
    except ValueError:
        pass

    without_ordinals = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text, flags=re.I)
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(without_ordinals, fmt).date().isoformat()
        except ValueError:
            continue

    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return "-".join(match.groups())

    LOGGER.debug("Unrecognized date format: %r", text)
    return ""


def limit_tags(tags: Sequence[str], source: str) -> list[str]:
    """Accept a tag list only if it is plausibly article-specific.

    A list longer than :data:`MAX_TAGS` is rejected entirely rather than
    truncated -- truncating would silently keep wrong values and look like a
    success.

    Args:
        tags: Candidate tags from one extraction level.
        source: Level name, used for logging.

    Returns:
        The cleaned tags, or ``[]`` if the list is implausibly long.
    """
    cleaned = dedupe(clean_text(tag) for tag in tags)
    cleaned = [tag for tag in cleaned if tag and tag.casefold() not in TAG_STOPWORDS]
    if not cleaned:
        return []
    if len(cleaned) > MAX_TAGS:
        LOGGER.warning(
            "Rejected %d tags from %s (limit %d) - looks like site-wide navigation",
            len(cleaned),
            source,
            MAX_TAGS,
        )
        return []
    return cleaned


# --------------------------------------------------------------------------- #
# Level 0: __NEXT_DATA__
# --------------------------------------------------------------------------- #


def find_next_data(soup: BeautifulSoup) -> dict[str, Any] | None:
    """Extract and decode the Next.js ``__NEXT_DATA__`` payload.

    Args:
        soup: Parsed article page.

    Returns:
        The decoded JSON object, or ``None`` if the script tag is missing or
        does not contain valid JSON.
    """
    script = soup.find("script", id="__NEXT_DATA__")
    if script is None:
        LOGGER.debug("No __NEXT_DATA__ script on page")
        return None

    raw = script.string if script.string is not None else script.get_text()
    if not raw or not raw.strip():
        LOGGER.warning("__NEXT_DATA__ script is empty")
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        LOGGER.warning("__NEXT_DATA__ is not valid JSON (%s)", exc)
        return None

    if not isinstance(payload, dict):
        LOGGER.warning("__NEXT_DATA__ decoded to %s, expected object", type(payload).__name__)
        return None
    return payload


def next_data_page(next_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return ``props.pageProps.page`` from a ``__NEXT_DATA__`` payload.

    Args:
        next_data: Decoded ``__NEXT_DATA__``, or ``None``.

    Returns:
        The page object, or ``None`` if the path does not exist.
    """
    if not next_data:
        return None
    page = dig(next_data, "props", "pageProps", "page")
    if not isinstance(page, dict):
        LOGGER.debug("__NEXT_DATA__ present but props.pageProps.page is missing")
        return None
    return page


def tags_from_next_data(next_data: dict[str, Any] | None) -> list[str]:
    """Read the article tags from the Next.js payload.

    The article taxonomy is ``page.category.tag`` (the main category) plus the
    ``tag`` field of every entry in ``page.subCategories``.  This is the only
    place on the page where article-specific tags exist.

    Args:
        next_data: Decoded ``__NEXT_DATA__``, or ``None``.

    Returns:
        Tags in ``[category, *subcategories]`` order; ``[]`` when unavailable.
    """
    page = next_data_page(next_data)
    if page is None:
        return []

    tags: list[str] = []

    category = page.get("category")
    if isinstance(category, dict):
        tags.append(clean_text(category.get("tag")))
    elif isinstance(category, str):
        tags.append(clean_text(category))

    subcategories = page.get("subCategories")
    if isinstance(subcategories, list):
        for entry in subcategories:
            if isinstance(entry, dict):
                tags.append(clean_text(entry.get("tag")))
            elif isinstance(entry, str):
                tags.append(clean_text(entry))

    return [tag for tag in tags if tag]


def field_from_next_data(next_data: dict[str, Any] | None, keys: Sequence[str]) -> str:
    """Best-effort scalar lookup in ``props.pageProps.page``.

    Used only as a last resort for fields the other levels failed to provide;
    unknown key names simply yield ``""``.

    Args:
        next_data: Decoded ``__NEXT_DATA__``, or ``None``.
        keys: Candidate field names, tried in order.

    Returns:
        The first non-empty string value found.
    """
    page = next_data_page(next_data)
    if page is None:
        return ""
    for key in keys:
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return clean_text(value)
        if isinstance(value, dict):
            for nested in ("name", "fullName", "title"):
                nested_value = value.get(nested)
                if isinstance(nested_value, str) and nested_value.strip():
                    return clean_text(nested_value)
    return ""


def listing_blogs(next_data: dict[str, Any] | None) -> list[Any] | None:
    """Return ``props.pageProps.blogs`` from a listing page payload.

    This array holds exactly the articles of the requested listing page (12 on
    the DataCamp blog) and nothing else -- unlike the markup, which also renders
    ``mostRecentBlogs`` and promoted posts from other sections.

    Args:
        next_data: Decoded ``__NEXT_DATA__``, or ``None``.

    Returns:
        The raw ``blogs`` list, or ``None`` if the path does not exist.
    """
    if not next_data:
        return None
    blogs = dig(next_data, "props", "pageProps", "blogs")
    if not isinstance(blogs, list):
        LOGGER.debug("__NEXT_DATA__ present but props.pageProps.blogs is missing")
        return None
    return blogs


def article_links_from_blogs(
    next_data: dict[str, Any] | None, base: str = BASE_URL
) -> list[str]:
    """Build article URLs from ``props.pageProps.blogs[].slug``.

    Authoritative link source for a listing page: one entry per article of that
    page, in the order the site lists them.

    Args:
        next_data: Decoded ``__NEXT_DATA__``, or ``None``.
        base: Site root.

    Returns:
        Canonical article URLs, deduplicated, in listing order.
    """
    blogs = listing_blogs(next_data)
    if not blogs:
        return []

    links: list[str] = []
    for entry in blogs:
        slug = entry.get("slug") if isinstance(entry, dict) else None
        if not isinstance(slug, str) or not slug.strip():
            LOGGER.debug("Skipping blogs[] entry without a usable slug")
            continue
        url = canonical_url(f"/blog/{slug.strip().strip('/')}", base)
        if url and is_article_url(url, base):
            links.append(url)
        else:
            LOGGER.debug("Skipping non-article slug %r from blogs[]", slug)
    return dedupe(links)


def listing_stats(next_data: dict[str, Any] | None) -> tuple[int | None, int | None]:
    """Read the pagination facts of a listing page.

    Args:
        next_data: Decoded ``__NEXT_DATA__``, or ``None``.

    Returns:
        A ``(count, per_page)`` pair, where ``count`` is the total number of
        articles on the blog (``props.pageProps.count``) and ``per_page`` is the
        size of this page's ``blogs`` array.  Either element is ``None`` when
        the payload does not provide a usable value.
    """
    raw_count = dig(next_data, "props", "pageProps", "count")
    count = raw_count if isinstance(raw_count, int) and not isinstance(raw_count, bool) else None
    if count is not None and count <= 0:
        count = None

    blogs = listing_blogs(next_data)
    per_page = len(blogs) if blogs else None
    return count, per_page


def total_listing_pages(count: int | None, per_page: int | None) -> int | None:
    """Compute how many listing pages the blog has.

    Args:
        count: Total number of articles, or ``None``.
        per_page: Articles per listing page, or ``None``.

    Returns:
        ``ceil(count / per_page)``, or ``None`` if either input is unusable.
    """
    if not count or not per_page or per_page <= 0:
        return None
    return math.ceil(count / per_page)


def article_links_from_next_data(
    next_data: dict[str, Any] | None, base: str = BASE_URL
) -> list[str]:
    """Recover article links by walking the whole Next.js payload.

    Last-resort fallback for a listing whose payload does not expose
    ``props.pageProps.blogs`` and whose markup is fully client-rendered.  Only
    objects carrying both a ``slug`` and a ``title`` are considered, and every
    candidate is validated with :func:`is_article_url`.  The walk is
    indiscriminate -- it also picks up sidebar and "most recent" sections -- so
    it ranks below both other sources.

    Args:
        next_data: Decoded ``__NEXT_DATA__``, or ``None``.
        base: Site root.

    Returns:
        Canonical article URLs in document order.
    """
    if not next_data:
        return []

    found: list[str] = []

    def walk(node: Any, depth: int) -> None:
        if depth > 10:
            return
        if isinstance(node, dict):
            slug = node.get("slug")
            title = node.get("title")
            if isinstance(slug, str) and slug.strip() and isinstance(title, str) and title.strip():
                url = canonical_url(f"/blog/{slug.strip().strip('/')}", base)
                if url and is_article_url(url, base):
                    found.append(url)
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, depth + 1)

    walk(next_data, 0)
    return dedupe(found)


# --------------------------------------------------------------------------- #
# Level 1: JSON-LD
# --------------------------------------------------------------------------- #


def iter_jsonld_objects(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Collect every JSON-LD object on the page, flattening ``@graph``.

    Args:
        soup: Parsed article page.

    Returns:
        A flat list of JSON-LD dictionaries; malformed blocks are skipped.
    """
    objects: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string if script.string is not None else script.get_text()
        if not raw or not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.debug("Skipping malformed JSON-LD block")
            continue
        queue = payload if isinstance(payload, list) else [payload]
        for item in queue:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                objects.extend(node for node in graph if isinstance(node, dict))
            else:
                objects.append(item)
    return objects


def is_article_type(node_type: Any) -> bool:
    """Check whether a JSON-LD ``@type`` describes an article.

    A substring match on ``"article"`` alone is not enough: schema.org calls the
    blog form ``BlogPosting``, which is what the DataCamp blog emits, so those
    names are listed explicitly in :data:`JSONLD_ARTICLE_TYPES`.  Fully
    qualified types (``https://schema.org/BlogPosting``) are reduced to their
    last path segment first.

    Args:
        node_type: Raw ``@type`` value: a string, a list of strings, or junk.

    Returns:
        ``True`` if any of the declared types is article-like.
    """
    types = node_type if isinstance(node_type, list) else [node_type]
    for value in types:
        if not isinstance(value, str):
            continue
        name = value.rsplit("/", 1)[-1].strip().casefold()
        if "article" in name or name in JSONLD_ARTICLE_TYPES:
            return True
    return False


def jsonld_article(soup: BeautifulSoup) -> dict[str, Any] | None:
    """Return the JSON-LD node describing the article, if present.

    Args:
        soup: Parsed article page.

    Returns:
        The first ``Article``-like node, or ``None``.
    """
    for node in iter_jsonld_objects(soup):
        if is_article_type(node.get("@type", "")):
            return node
    return None


def jsonld_author(node: dict[str, Any]) -> str:
    """Extract an author name from a JSON-LD article node.

    Args:
        node: JSON-LD article object.

    Returns:
        Author name, or ``""``.
    """
    author = node.get("author")
    if isinstance(author, list):
        author = author[0] if author else None
    if isinstance(author, dict):
        return clean_text(author.get("name"))
    return clean_text(author)


def tags_from_jsonld(soup: BeautifulSoup) -> list[str]:
    """Read tags from JSON-LD ``keywords``/``articleSection``/``about``.

    Args:
        soup: Parsed article page.

    Returns:
        Candidate tags before the size check.
    """
    node = jsonld_article(soup)
    if node is None:
        return []

    tags: list[str] = []
    for key in ("keywords", "articleSection", "about"):
        value = node.get(key)
        if isinstance(value, str):
            tags.extend(part for part in re.split(r"\s*,\s*", value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    tags.append(item)
                elif isinstance(item, dict):
                    tags.append(clean_text(item.get("name")))
    return [tag for tag in (clean_text(tag) for tag in tags) if tag]


# --------------------------------------------------------------------------- #
# Level 2: <meta> tags
# --------------------------------------------------------------------------- #


def meta_content(soup: BeautifulSoup, *names: str) -> str:
    """Return the content of the first matching ``<meta>`` tag.

    Args:
        soup: Parsed page.
        *names: ``property``/``name`` values to try, in order.

    Returns:
        The tag content, or ``""``.
    """
    for name in names:
        for attr in ("property", "name", "itemprop"):
            tag = soup.find("meta", attrs={attr: name})
            if tag is not None:
                content = clean_text(tag.get("content"))
                if content:
                    return content
    return ""


def tags_from_meta(soup: BeautifulSoup) -> list[str]:
    """Read tags from ``article:tag`` and ``keywords`` meta tags.

    Args:
        soup: Parsed article page.

    Returns:
        Candidate tags before the size check.
    """
    tags: list[str] = []
    for attr in ("property", "name"):
        for tag in soup.find_all("meta", attrs={attr: "article:tag"}):
            tags.append(clean_text(tag.get("content")))
    if not tags:
        keywords = meta_content(soup, "keywords", "article:section")
        if keywords:
            tags.extend(re.split(r"\s*,\s*", keywords))
    return [tag for tag in (clean_text(tag) for tag in tags) if tag]


# --------------------------------------------------------------------------- #
# Level 3: HTML heuristics
# --------------------------------------------------------------------------- #

#: Containers that are never part of the article body.  ``header`` is kept on
#: purpose: article tags often sit next to the title inside it.
NOISE_SELECTORS = (
    "nav",
    "footer",
    "aside",
    "[role=navigation]",
    "[role=contentinfo]",
    "[class*=sidebar]",
    "[class*=Sidebar]",
    "[class*=menu]",
    "[class*=Menu]",
    "[class*=related]",
    "[class*=Related]",
    "[class*=recommended]",
    "[class*=Recommended]",
    "[class*=footer]",
    "[class*=Footer]",
)

#: Containers that announce themselves as the article's tag block.
TAG_CONTAINER_SELECTORS = (
    "[class*=article-tag]",
    "[class*=post-tag]",
    "[class*=entry-tag]",
    "[class*=article-categor]",
    "[class*=post-categor]",
    "[data-testid*=tag]",
    "[data-testid*=categor]",
    "[class*=taxonomy]",
)

TAXONOMY_PATH_MARKERS = ("/category/", "/categories/", "/tag/", "/tags/", "/topic/", "/topics/")


def article_scope(soup: BeautifulSoup) -> Tag | None:
    """Return the narrowest element that plausibly contains the article.

    Args:
        soup: Parsed article page.

    Returns:
        A detached copy of the article container with navigation chrome
        removed, or ``None`` if the document is empty.
    """
    scope = (
        soup.find("article")
        or soup.select_one("[itemprop=articleBody]")
        or soup.find("main")
        or soup.body
    )
    if scope is None:
        return None

    scope = copy.copy(scope)
    for selector in NOISE_SELECTORS:
        try:
            matches = scope.select(selector)
        except Exception:  # pragma: no cover - invalid selector guard
            continue
        for element in matches:
            element.decompose()
    return scope


def taxonomy_links(node: Tag) -> list[Tag]:
    """Find links that point at a category/tag listing.

    Args:
        node: Element to search within.

    Returns:
        Matching ``<a>`` elements in document order.
    """
    links: list[Tag] = []
    for link in node.find_all("a", href=True):
        path = urlsplit(link["href"]).path.casefold()
        if not path.endswith("/"):
            path += "/"
        if any(marker in path for marker in TAXONOMY_PATH_MARKERS):
            links.append(link)
    return links


def link_labels(links: Iterable[Tag]) -> list[str]:
    """Turn link elements into clean tag labels.

    ``get_text(" ")`` is used so icon and screen-reader spans do not fuse into
    strings like ``See MoreRight Arrow``.

    Args:
        links: Link elements.

    Returns:
        Deduplicated labels with boilerplate removed.
    """
    labels: list[str] = []
    for link in links:
        label = clean_text(link.get_text(" ", strip=True))
        if not label or len(label) > 60:
            continue
        if label.casefold() in TAG_STOPWORDS:
            continue
        if label.isdigit():
            continue
        labels.append(label)
    return dedupe(labels)


def tags_from_html(soup: BeautifulSoup) -> list[str]:
    """Extract article tags from the rendered markup.

    Last-resort level.  The search is confined to the article container, and
    candidate groups larger than :data:`MAX_TAGS` are dropped so a category
    menu that survived scoping still cannot win.

    Args:
        soup: Parsed article page.

    Returns:
        Candidate tags before the final size check.
    """
    scope = article_scope(soup)
    if scope is None:
        return []

    order = {id(element): index for index, element in enumerate(scope.descendants)}
    heading = scope.find("h1")
    anchor = order.get(id(heading), 0) if heading is not None else 0

    def position(element: Tag) -> int:
        return order.get(id(element), len(order))

    candidates: list[tuple[int, list[str]]] = []

    # Prefer containers that declare themselves as the tag block.
    for selector in TAG_CONTAINER_SELECTORS:
        try:
            matches = scope.select(selector)
        except Exception:  # pragma: no cover - invalid selector guard
            continue
        for container in matches:
            labels = link_labels(taxonomy_links(container))
            if labels and len(labels) <= MAX_TAGS:
                candidates.append((abs(position(container) - anchor), labels))

    # Otherwise group taxonomy links by their parent: the article's own tags
    # form one small group, the site menu forms one huge group.
    if not candidates:
        groups: dict[int, tuple[Tag, list[Tag]]] = {}
        for link in taxonomy_links(scope):
            parent = link.parent
            if parent is None:
                continue
            groups.setdefault(id(parent), (parent, []))[1].append(link)
        for parent, links in groups.values():
            if len(links) > MAX_TAGS:
                LOGGER.debug("Ignoring group of %d taxonomy links (site menu)", len(links))
                continue
            labels = link_labels(links)
            if labels and len(labels) <= MAX_TAGS:
                candidates.append((abs(position(parent) - anchor), labels))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


# --------------------------------------------------------------------------- #
# Page parsing
# --------------------------------------------------------------------------- #


def extract_tags(soup: BeautifulSoup, next_data: dict[str, Any] | None) -> tuple[list[str], str]:
    """Run the four tag levels in priority order.

    Args:
        soup: Parsed article page.
        next_data: Decoded ``__NEXT_DATA__``, or ``None``.

    Returns:
        A ``(tags, source)`` pair; ``source`` is ``"none"`` when nothing was
        found, in which case the column is left empty rather than filled with
        navigation links.
    """
    levels = (
        ("__NEXT_DATA__", lambda: tags_from_next_data(next_data)),
        ("json-ld", lambda: tags_from_jsonld(soup)),
        ("meta", lambda: tags_from_meta(soup)),
        ("html", lambda: tags_from_html(soup)),
    )
    for source, extractor in levels:
        tags = limit_tags(extractor(), source)
        if tags:
            return tags, source
    return [], "none"


def extract_excerpt(soup: BeautifulSoup) -> str:
    """Find a short description of the article.

    The article body is never stored; this is the meta description or, failing
    that, the first paragraph.

    Args:
        soup: Parsed article page.

    Returns:
        A short excerpt, truncated to 500 characters.
    """
    node = jsonld_article(soup)
    excerpt = clean_text(node.get("description")) if node else ""
    if not excerpt:
        excerpt = meta_content(soup, "og:description", "description", "twitter:description")
    if not excerpt:
        scope = article_scope(soup)
        if scope is not None:
            for paragraph in scope.find_all("p"):
                text = clean_text(paragraph.get_text(" ", strip=True))
                if len(text) >= 40:
                    excerpt = text
                    break
    return excerpt[:500]


def parse_article_page(html: str, url: str) -> dict[str, str]:
    """Extract metadata for a single article page.

    Args:
        html: Raw HTML of the article page.
        url: Canonical URL of the article.

    Returns:
        A row dictionary matching :data:`COLUMNS`.  Missing values are empty
        strings, never ``NaN``.
    """
    soup = make_soup(html)
    next_data = find_next_data(soup)
    node = jsonld_article(soup) or {}

    title = clean_text(node.get("headline")) or meta_content(soup, "og:title", "twitter:title")
    if not title:
        heading = soup.find("h1")
        title = clean_text(heading.get_text(" ", strip=True)) if heading else ""
    if not title:
        title = field_from_next_data(next_data, ("title", "headline", "name"))

    raw_date = clean_text(node.get("datePublished")) or meta_content(
        soup, "article:published_time", "og:article:published_time", "datePublished"
    )
    if not raw_date:
        time_tag = soup.find("time")
        if time_tag is not None:
            raw_date = clean_text(time_tag.get("datetime")) or clean_text(time_tag.get_text())
    if not raw_date:
        raw_date = field_from_next_data(
            next_data, ("publishedAt", "publishDate", "datePublished", "date")
        )
    published_date = parse_date(raw_date)

    author = jsonld_author(node) or meta_content(soup, "article:author", "author")
    if not author or author.startswith("http"):
        author_tag = soup.select_one("[rel=author]") or soup.select_one("[itemprop=author]")
        if author_tag is not None:
            author = clean_text(author_tag.get_text(" ", strip=True))
    if not author:
        author = field_from_next_data(next_data, ("author", "authorName"))

    tags, tag_source = extract_tags(soup, next_data)

    if not title:
        LOGGER.warning("No title found for %s - page structure may have changed", url)
    if not tags:
        LOGGER.warning("No tags found for %s (all four levels empty)", url)
    else:
        LOGGER.debug("Tags for %s from %s: %s", url, tag_source, tags)

    return {
        "title": title,
        "url": url,
        "published_date": published_date,
        "author": author,
        "tags": TAG_SEPARATOR.join(tags),
        "excerpt": extract_excerpt(soup),
        "scraped_at": datetime.now(timezone.utc).date().isoformat(),
    }


def links_from_markup(soup: BeautifulSoup, page_url: str, base: str = BASE_URL) -> list[str]:
    """Collect ``/blog/<slug>`` links from the rendered markup.

    Args:
        soup: Parsed listing page.
        page_url: URL the HTML came from, used to resolve relative links.
        base: Site root.

    Returns:
        Canonical article URLs in document order.
    """
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        url = canonical_url(anchor["href"], page_url)
        if url and is_article_url(url, base):
            links.append(url)
    return links


def listing_article_links(
    soup: BeautifulSoup, page_url: str, base: str = BASE_URL
) -> list[str]:
    """Collect the article URLs of a listing page from an already parsed tree.

    Three ordered sources, first non-empty wins:

        0. ``__NEXT_DATA__`` -> ``props.pageProps.blogs[].slug`` (authoritative:
           exactly this page's articles)
        1. ``<a href="/blog/...">`` in the markup -- mixes several page sections
           together, so it is only a fallback
        2. a full walk of ``__NEXT_DATA__`` for ``slug``/``title`` objects

    Args:
        soup: Parsed listing page.
        page_url: URL the HTML came from, used to resolve relative links.
        base: Site root.

    Returns:
        Canonical article URLs, deduplicated.
    """
    next_data = find_next_data(soup)

    links = article_links_from_blogs(next_data, base)
    if links:
        LOGGER.debug("%s: %d links from props.pageProps.blogs", page_url, len(links))
        return links

    LOGGER.warning(
        "No props.pageProps.blogs on %s - falling back to the markup", page_url
    )
    links = dedupe(links_from_markup(soup, page_url, base))
    if links:
        return links

    LOGGER.warning("No article links in the markup of %s either - walking __NEXT_DATA__", page_url)
    links = article_links_from_next_data(next_data, base)
    if links:
        LOGGER.info("Recovered %d links by walking __NEXT_DATA__", len(links))
    return links


def extract_article_links(html: str, page_url: str, base: str = BASE_URL) -> list[str]:
    """Collect article URLs from the raw HTML of a blog listing page.

    Args:
        html: Raw HTML of the listing page.
        page_url: URL the HTML came from, used to resolve relative links.
        base: Site root.

    Returns:
        Canonical article URLs, deduplicated.
    """
    return listing_article_links(make_soup(html), page_url, base)


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #


class RobotsPolicy:
    """``robots.txt`` rules for the crawl."""

    def __init__(self, parser: RobotFileParser, user_agent: str, crawl_delay: float | None):
        """Initialize the policy.

        Args:
            parser: A parser that has already consumed ``robots.txt``.
            user_agent: User agent the rules are evaluated against.
            crawl_delay: ``Crawl-delay`` in seconds, if declared.
        """
        self.parser = parser
        self.user_agent = user_agent
        self.crawl_delay = crawl_delay

    @classmethod
    def load(cls, session: requests.Session, base: str = BASE_URL) -> "RobotsPolicy":
        """Download and parse ``robots.txt``.

        Args:
            session: HTTP session to use.
            base: Site root.

        Returns:
            The parsed policy.

        Raises:
            RobotsUnavailableError: If the file cannot be read, in which case
                crawling must not proceed.
        """
        robots_url = urljoin(base, "/robots.txt")
        try:
            response = session.get(robots_url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            raise RobotsUnavailableError(f"Could not download {robots_url}: {exc}") from exc
        if response.status_code != 200:
            raise RobotsUnavailableError(f"{robots_url} returned HTTP {response.status_code}")

        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        delay = parser.crawl_delay(USER_AGENT)
        LOGGER.info("robots.txt loaded; declared Crawl-delay: %s", delay)
        return cls(parser, USER_AGENT, float(delay) if delay else None)

    def can_fetch(self, url: str) -> bool:
        """Check a single URL against the rules.

        Args:
            url: Absolute URL.

        Returns:
            ``True`` if fetching is allowed.
        """
        return self.parser.can_fetch(self.user_agent, url)


class Throttle:
    """Sleeps between requests to keep the crawl polite."""

    def __init__(self, delay_min: float, delay_max: float):
        """Initialize the throttle.

        Args:
            delay_min: Lower bound in seconds; raised to :data:`MIN_ALLOWED_DELAY`.
            delay_max: Upper bound in seconds.
        """
        self.delay_min = max(delay_min, MIN_ALLOWED_DELAY)
        self.delay_max = max(delay_max, self.delay_min)
        self._first = True

    def wait(self) -> None:
        """Sleep before the next request (no-op before the first one)."""
        if self._first:
            self._first = False
            return
        time.sleep(random.uniform(self.delay_min, self.delay_max))


def fetch(session: requests.Session, url: str, throttle: Throttle, robots: RobotsPolicy) -> str | None:
    """Fetch a page with retries, backoff and a robots check.

    Args:
        session: HTTP session.
        url: Absolute URL to fetch.
        throttle: Delay controller.
        robots: Policy consulted before every request.

    Returns:
        Response text, or ``None`` if the page is disallowed, missing or
        unavailable after all retries.
    """
    if not robots.can_fetch(url):
        LOGGER.warning("robots.txt disallows %s - skipping", url)
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        throttle.wait()
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
        except requests.Timeout:
            LOGGER.warning("Timeout on %s (attempt %d/%d)", url, attempt, MAX_RETRIES)
        except requests.RequestException as exc:
            LOGGER.warning("Request failed for %s: %s (attempt %d/%d)", url, exc, attempt, MAX_RETRIES)
        else:
            if response.status_code == 404:
                LOGGER.warning("404 Not Found: %s - skipping", url)
                return None
            if response.status_code == 429 or response.status_code >= 500:
                LOGGER.warning(
                    "HTTP %d on %s (attempt %d/%d)", response.status_code, url, attempt, MAX_RETRIES
                )
            elif response.status_code >= 400:
                LOGGER.warning("HTTP %d on %s - skipping", response.status_code, url)
                return None
            else:
                return response.text

        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_SECONDS * attempt)

    LOGGER.error("Giving up on %s after %d attempts", url, MAX_RETRIES)
    return None


def listing_url(page: int, base: str = BASE_URL) -> str:
    """Build the URL of a blog listing page.

    Pagination is path-based: page 1 is ``/blog`` and every later page is
    ``/blog/page/N``.  The ``?page=N`` query form does exist but is ignored by
    the site, which silently serves page 1 for it.

    Args:
        page: 1-based page number.
        base: Site root.

    Returns:
        Absolute listing URL.
    """
    if page <= 1:
        return urljoin(base, LISTING_PATH)
    return urljoin(base, f"{LISTING_PAGE_PATH}/{page}")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def read_existing(path: Path) -> list[dict[str, str]]:
    """Load previously scraped rows, if the output file exists.

    Args:
        path: CSV path.

    Returns:
        Existing rows normalized to :data:`COLUMNS`; ``[]`` when absent or
        unreadable.
    """
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [
                {column: clean_text(row.get(column, "")) for column in COLUMNS}
                for row in csv.DictReader(handle)
            ]
    except (OSError, csv.Error) as exc:
        LOGGER.warning("Could not read %s (%s) - starting fresh", path, exc)
        return []


def merge_rows(existing: Sequence[dict[str, str]], fresh: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Merge new rows into old ones, deduplicating by URL.

    Makes repeated runs idempotent: a re-scraped article replaces its previous
    record instead of being appended again.

    Args:
        existing: Rows already on disk.
        fresh: Rows from this run; these win on conflict.

    Returns:
        Merged rows sorted by publication date (newest first), then URL.
    """
    merged: dict[str, dict[str, str]] = {row["url"]: row for row in existing if row.get("url")}
    for row in fresh:
        if row.get("url"):
            merged[row["url"]] = row
    return sorted(merged.values(), key=lambda row: (row.get("published_date", ""), row["url"]), reverse=True)


def validate_rows(rows: Sequence[dict[str, str]]) -> None:
    """Sanity-check rows before writing them.

    Args:
        rows: Rows about to be written.

    Raises:
        AssertionError: If the schema, uniqueness or tag-count invariants fail.
    """
    assert rows, "Nothing to write - no articles were collected"
    urls = [row["url"] for row in rows]
    assert len(urls) == len(set(urls)), "Duplicate URLs in output"
    for row in rows:
        assert set(row) == set(COLUMNS), f"Unexpected columns: {sorted(set(row) ^ set(COLUMNS))}"
        assert row["url"], "Empty url"
        assert row["title"], f"Empty title for {row['url']}"
        tags = [tag for tag in row["tags"].split(TAG_SEPARATOR) if tag]
        assert len(tags) <= MAX_TAGS, (
            f"{row['url']} has {len(tags)} tags (limit {MAX_TAGS}) - "
            "site navigation probably leaked into the tags column"
        )


def write_csv(rows: Sequence[dict[str, str]], path: Path) -> None:
    """Write rows to CSV, creating parent directories as needed.

    Args:
        rows: Validated rows.
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %d rows to %s", len(rows), path)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def collect_article_urls(
    session: requests.Session,
    robots: RobotsPolicy,
    throttle: Throttle,
    pages: int,
    start_page: int,
    auto_pages: bool = False,
) -> list[str]:
    """Walk the listing pagination and gather article URLs.

    Args:
        session: HTTP session.
        robots: Robots policy.
        throttle: Delay controller.
        pages: How many listing pages to visit.
        start_page: 1-based page to start from.
        auto_pages: Derive the last page from ``props.pageProps.count`` and the
            size of the first page's ``blogs`` array instead of stopping at
            ``pages``.  Falls back to ``pages`` if the payload lacks either
            value.

    Returns:
        Deduplicated article URLs.
    """
    urls: list[str] = []
    seen: set[str] = set()
    page = start_page
    # --auto-pages needs the first page fetched: it carries count and per-page.
    last_page = start_page + (max(pages, 1) if auto_pages else pages) - 1
    page_count_known = False

    while page <= last_page:
        url = listing_url(page)
        html = fetch(session, url, throttle, robots)
        if html is None:
            LOGGER.warning("Listing page %d unavailable - stopping pagination", page)
            break

        soup = make_soup(html)
        if auto_pages and page == start_page:
            count, per_page = listing_stats(find_next_data(soup))
            total = total_listing_pages(count, per_page)
            if total is None:
                LOGGER.warning(
                    "--auto-pages: count=%s per_page=%s unusable - keeping --pages %d",
                    count,
                    per_page,
                    pages,
                )
            else:
                last_page = total
                page_count_known = True
                LOGGER.info(
                    "--auto-pages: %d articles / %d per page = %d pages (visiting %d-%d)",
                    count,
                    per_page,
                    total,
                    start_page,
                    last_page,
                )

        found = listing_article_links(soup, url)
        new = [item for item in found if item not in seen]
        LOGGER.info("Page %d: %d links, %d new", page, len(found), len(new))
        if not found:
            LOGGER.warning("Page %d has no article links at all - stopping pagination", page)
            break
        if not new:
            # Without a known page count, a repeat page is the only end signal.
            if not page_count_known:
                LOGGER.info("Page %d added nothing new - end of listing or layout change", page)
                break
            LOGGER.warning("Page %d repeats already-seen articles - continuing", page)
        seen.update(new)
        urls.extend(new)
        page += 1

    return urls


def scrape(
    pages: int,
    start_page: int,
    limit: int | None,
    fetch_details: bool,
    delay_min: float,
    delay_max: float,
    auto_pages: bool = False,
) -> list[dict[str, str]]:
    """Run the full scrape.

    Args:
        pages: Number of listing pages to visit.
        start_page: 1-based listing page to start from.
        limit: Optional cap on the number of articles.
        fetch_details: Whether to open each article page.
        delay_min: Minimum delay between requests, in seconds.
        delay_max: Maximum delay between requests, in seconds.
        auto_pages: Derive the number of listing pages from the first page's
            ``__NEXT_DATA__`` instead of using ``pages``.

    Returns:
        Collected rows.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})

    robots = RobotsPolicy.load(session)
    throttle = Throttle(delay_min, delay_max)
    if robots.crawl_delay and robots.crawl_delay > throttle.delay_min:
        LOGGER.info("Honouring Crawl-delay of %.1fs", robots.crawl_delay)
        throttle.delay_min = robots.crawl_delay
        throttle.delay_max = max(robots.crawl_delay, throttle.delay_max)

    urls = collect_article_urls(session, robots, throttle, pages, start_page, auto_pages)
    if limit is not None:
        urls = urls[:limit]
    LOGGER.info("Collected %d article URLs", len(urls))

    rows: list[dict[str, str]] = []
    today = datetime.now(timezone.utc).date().isoformat()
    for index, url in enumerate(urls, start=1):
        if not fetch_details:
            rows.append(
                {
                    "title": "",
                    "url": url,
                    "published_date": "",
                    "author": "",
                    "tags": "",
                    "excerpt": "",
                    "scraped_at": today,
                }
            )
            continue
        LOGGER.info("[%d/%d] %s", index, len(urls), url)
        html = fetch(session, url, throttle, robots)
        if html is None:
            continue
        rows.append(parse_article_page(html, url))
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description="Scrape DataCamp blog article metadata.")
    parser.add_argument("--pages", type=int, default=2, help="listing pages to visit (default: 2)")
    parser.add_argument("--start-page", type=int, default=1, help="first listing page (default: 1)")
    parser.add_argument(
        "--auto-pages",
        action="store_true",
        help=(
            "derive the last listing page from props.pageProps.count and the "
            "size of the first page's blogs array, ignoring --pages"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="max articles to fetch")
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="collect URLs only, do not open each article page",
    )
    parser.add_argument("--delay-min", type=float, default=DEFAULT_DELAY_MIN)
    parser.add_argument("--delay-max", type=float, default=DEFAULT_DELAY_MAX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output CSV path")
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
        fresh = scrape(
            pages=args.pages,
            start_page=args.start_page,
            limit=args.limit,
            fetch_details=not args.no_details,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
            auto_pages=args.auto_pages,
        )
    except RobotsUnavailableError as exc:
        LOGGER.error("Refusing to crawl: %s", exc)
        return 2

    if not fresh:
        LOGGER.error("No articles collected - check the listing selectors with --verbose")
        return 1

    rows = merge_rows(read_existing(args.output), fresh)
    validate_rows(rows)
    write_csv(rows, args.output)

    tagged = sum(1 for row in rows if row["tags"])
    LOGGER.info("Rows with tags: %d/%d", tagged, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
