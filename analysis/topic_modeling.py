#!/usr/bin/env python3
"""TF-IDF + LDA topic modelling over the cleaned DataCamp article dataset.

The corpus is ``title + excerpt`` from ``data/processed/articles_clean.csv``
(there is no full article text, by design -- only metadata).  The pipeline is:

    preprocess -> TF-IDF -> LDA for several k -> pick k by coherence
              -> label topics -> cross-check against tags -> write artifacts

**Why scikit-learn and not gensim.**  ``gensim`` is the more classic LDA
implementation and ships ``CoherenceModel``, but it pins older ``scipy``
internals (``scipy.linalg.triu``) and breaks on the NumPy 2 / Python 3.12 stack
this project targets.  scikit-learn is already required for TF-IDF, so
``LatentDirichletAllocation`` keeps the dependency surface at one library and
runs everywhere.  The one thing gensim would have given us for free -- topic
coherence -- is implemented here as :func:`umass_coherence`, which needs only
document-term co-occurrence counts.

**TF-IDF as LDA input.**  LDA is formally a generative model over counts, and
feeding it TF-IDF weights is a documented approximation rather than textbook
usage.  It is what the analysis asks for and it works well on short text, where
IDF is what stops ``data`` and ``learn`` from dominating every topic.  Pass
``--vectorizer count`` to fit the textbook variant instead; the rest of the
pipeline is identical.

Outputs (all deterministic -- ``random_state`` is fixed, so re-running
overwrites with byte-identical files):

    data/processed/topics.json               topics, keywords, tag profile, edges
    data/processed/articles_with_topics.csv  articles + dominant topic
    analysis/topic_modeling_report.md        human-readable topic descriptions

``topics.json`` is shaped for the next stage (``viz/``): ``topics[].topic_id``
are the future networkx nodes, ``edges`` are ready-made weighted edges from
keyword similarity, and the CSV's ``topic_second_id`` supports building
co-assignment edges instead.

Usage:
    python analysis/topic_modeling.py
    python analysis/topic_modeling.py --topics 12          # skip the search
    python analysis/topic_modeling.py --topic-range 6 24 2 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Importable both as `python analysis/topic_modeling.py` and from a notebook
# whose working directory is the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_data import load_clean, split_tags  # noqa: E402
from text_preprocessing import join_tokens, preprocess_documents  # noqa: E402

DEFAULT_INPUT = Path("data/processed/articles_clean.csv")
DEFAULT_TOPICS_JSON = Path("data/processed/topics.json")
DEFAULT_ARTICLES_CSV = Path("data/processed/articles_with_topics.csv")
DEFAULT_REPORT = Path("analysis/topic_modeling_report.md")

#: Fixed everywhere so two runs on the same input give the same topics.
RANDOM_STATE = 42

#: Candidate topic counts: (start, stop, step) for the coherence search.
DEFAULT_TOPIC_RANGE = (8, 22, 2)

#: Keywords reported per topic.
TOP_KEYWORDS = 10

#: Tags reported per topic in the cross-check.
TOP_TAGS_PER_TOPIC = 5

#: Example articles quoted per topic in the report.
EXAMPLES_PER_TOPIC = 3

#: Unigrams and bigrams -- `machine learning` and `data engineering` are single
#: concepts and read far better in a topic label than two separate words.
NGRAM_RANGE = (1, 2)

#: A term must appear in at least this many documents to enter the vocabulary.
DEFAULT_MIN_DF = 5

#: A term appearing in more than this share of documents is corpus-wide noise.
DEFAULT_MAX_DF = 0.5

#: Vocabulary cap; the corpus is short so this is generous in practice.
DEFAULT_MAX_FEATURES = 5000

#: Below this many documents `min_df=5` would empty the vocabulary (tests,
#: notebook slices), so it is forced down to 1.
SMALL_CORPUS_THRESHOLD = 50

#: Cosine similarity below which a topic-topic edge is not worth a graph edge.
EDGE_THRESHOLD = 0.10

LOGGER = logging.getLogger("topic_modeling")


class DatasetError(ValueError):
    """Raised when the input dataset cannot be used for topic modelling."""


# --------------------------------------------------------------------------- #
# Topic labels
# --------------------------------------------------------------------------- #

#: Human-readable topic names keyed by the terms that trigger them.  A topic is
#: labelled by whichever rule accumulates the most keyword weight, so the label
#: follows the data instead of the topic's arbitrary index.  Rules are written
#: in lemma form (`statistic`, not `statistics`) because they are matched
#: against already-lemmatized keywords.
LABEL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Великі мовні моделі та генеративний AI",
        ("llm", "gpt", "chatgpt", "claude", "gemini", "llama", "prompt", "generative",
         "generative ai", "rag", "fine-tuning", "agent", "chatbot", "token", "mistral"),
    ),
    (
        "Глибоке навчання та нейромережі",
        ("deep learning", "neural", "neural network", "pytorch", "tensorflow", "keras",
         "transformer", "cnn", "rnn", "lstm", "gan", "embedding", "computer vision", "cv"),
    ),
    (
        "Машинне навчання: моделі та алгоритми",
        ("machine learning", "ml", "algorithm", "classification", "regression", "supervised",
         "unsupervised", "random forest", "xgboost", "feature", "prediction", "predictive",
         "scikit-learn", "overfitting", "hyperparameter"),
    ),
    (
        "SQL та бази даних",
        ("sql", "database", "query", "postgresql", "mysql", "join", "nosql", "mongodb",
         "table", "schema", "relational", "index", "sqlite", "oracle"),
    ),
    (
        "Хмарні платформи та Data Engineering",
        ("cloud", "aws", "azure", "gcp", "databricks", "snowflake", "pipeline", "etl",
         "warehouse", "spark", "docker", "kubernetes", "airflow", "data engineering",
         "engineer", "infrastructure", "lakehouse"),
    ),
    (
        "MLOps та впровадження моделей у продакшн",
        ("mlops", "deployment", "deploy", "production", "monitoring", "api", "serving",
         "versioning", "ci/cd", "scalable", "latency"),
    ),
    (
        "Python: мова та бібліотеки",
        ("python", "pandas", "numpy", "function", "list", "dictionary", "string", "loop",
         "syntax", "script", "library", "jupyter", "notebook", "package"),
    ),
    (
        "R та статистичний аналіз",
        ("r", "rstudio", "ggplot", "statistic", "statistical", "hypothesis", "probability",
         "distribution", "variance", "correlation", "significance", "sample"),
    ),
    (
        "Бізнес-аналітика та візуалізація даних",
        ("power bi", "powerbi", "tableau", "dashboard", "visualization", "visualisation",
         "chart", "excel", "spreadsheet", "report", "kpi", "looker", "plot", "graph"),
    ),
    (
        "Аналіз даних та робота аналітика",
        ("data analysis", "analyst", "analytics", "insight", "metric", "eda", "exploratory",
         "dataset", "cleaning", "data quality", "spreadsheet"),
    ),
    (
        "Кар'єра та навчання в Data Science",
        ("career", "job", "interview", "salary", "certification", "certificate", "course",
         "skill", "resume", "bootcamp", "hire", "hiring", "beginner", "portfolio",
         "roadmap", "student", "degree"),
    ),
    (
        "AI у бізнесі: стратегія та трансформація",
        ("business", "strategy", "company", "organization", "enterprise", "decision",
         "leadership", "upskilling", "team", "roi", "customer", "industry", "workflow",
         "productivity"),
    ),
    (
        "Етика, регулювання та майбутнє AI",
        ("ethic", "ethical", "regulation", "governance", "safety", "bias", "privacy",
         "risk", "future", "trend", "impact", "society", "policy", "responsible"),
    ),
    (
        "Обробка природної мови",
        ("nlp", "text", "language", "sentiment", "translation", "speech", "corpus",
         "tokenization", "named entity"),
    ),
    (
        "Інструменти, порівняння та практичні гайди",
        ("tool", "guide", "tutorial", "comparison", "alternative", "step", "example",
         "cheat sheet", "template", "tip"),
    ),
)


def label_topic(keywords: Sequence[tuple[str, float]]) -> tuple[str | None, float]:
    """Pick a human-readable name for a topic from its keywords.

    Each rule in :data:`LABEL_RULES` scores the sum of the weights of the
    keywords it matches; matching is whole-word and phrase-aware, so the rule
    ``ai`` matches the keyword ``ai`` and ``generative ai`` but not ``train``.

    Args:
        keywords: ``(term, weight)`` pairs for one topic, highest weight first.

    Returns:
        A ``(label, score)`` pair.  ``label`` is ``None`` when no rule matched,
        which lets the caller fall back to a keyword-derived name.
    """
    best_label: str | None = None
    best_score = 0.0

    for label, triggers in LABEL_RULES:
        score = sum(
            weight
            for term, weight in keywords
            if any(_phrase_in_term(trigger, term) for trigger in triggers)
        )
        if score > best_score:
            best_label, best_score = label, score

    return best_label, best_score


def _phrase_in_term(phrase: str, term: str) -> bool:
    """Check whether a rule phrase occurs as whole words inside a keyword.

    Args:
        phrase: Rule trigger, possibly multi-word (``machine learning``).
        term: Topic keyword, possibly a bigram (``machine learning model``).

    Returns:
        ``True`` if the phrase's words appear contiguously in the term.
    """
    phrase_words = phrase.split()
    term_words = term.split()
    span = len(phrase_words)
    return any(term_words[i : i + span] == phrase_words for i in range(len(term_words) - span + 1))


def fallback_label(topic_id: int, keywords: Sequence[tuple[str, float]]) -> str:
    """Name a topic from its own keywords when no rule matched.

    Args:
        topic_id: Zero-based topic index.
        keywords: ``(term, weight)`` pairs, highest weight first.

    Returns:
        A label such as ``Тема 7: sql, query, join``.
    """
    top = ", ".join(term for term, _ in keywords[:3])
    return f"Тема {topic_id}: {top}"


def assign_labels(topic_keywords: Sequence[Sequence[tuple[str, float]]]) -> list[str]:
    """Label every topic, keeping the labels distinct.

    Two topics can legitimately both look like "machine learning"; when that
    happens the runner-up gets its strongest keyword appended so the report
    never shows two identically named topics.

    Args:
        topic_keywords: Per-topic ``(term, weight)`` pairs.

    Returns:
        One label per topic, in topic order.
    """
    scored = [label_topic(keywords) for keywords in topic_keywords]
    # A rule may win in several topics; the topic where it scores highest keeps
    # the plain name, the others are disambiguated by their own top keyword.
    best_owner: dict[str, int] = {}
    for topic_id, (label, score) in enumerate(scored):
        if label is None:
            continue
        current = best_owner.get(label)
        if current is None or score > scored[current][1]:
            best_owner[label] = topic_id

    labels: list[str] = []
    seen: set[str] = set()
    for topic_id, (label, _) in enumerate(scored):
        keywords = topic_keywords[topic_id]
        if label is None:
            candidate = fallback_label(topic_id, keywords)
        elif best_owner[label] == topic_id:
            candidate = label
        else:
            candidate = f"{label} ({keywords[0][0]})" if keywords else f"{label} ({topic_id})"
        if candidate in seen:
            candidate = f"{candidate} [{topic_id}]"
        seen.add(candidate)
        labels.append(candidate)
    return labels


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def build_corpus(frame: pd.DataFrame) -> list[str]:
    """Join ``title`` and ``excerpt`` into one document per article.

    Args:
        frame: The cleaned dataset.

    Returns:
        Raw (not yet preprocessed) documents aligned with ``frame``.

    Raises:
        DatasetError: If either text column is missing.
    """
    missing = [column for column in ("title", "excerpt") if column not in frame.columns]
    if missing:
        raise DatasetError(f"input is missing columns: {', '.join(missing)}")
    return (frame["title"].astype(str) + ". " + frame["excerpt"].astype(str)).tolist()


def effective_min_df(n_documents: int, min_df: int) -> int:
    """Lower ``min_df`` on corpora too small for the production setting.

    Args:
        n_documents: Number of documents.
        min_df: Configured minimum document frequency.

    Returns:
        ``min_df`` for a full corpus, ``1`` for a small one.
    """
    return min_df if n_documents >= SMALL_CORPUS_THRESHOLD else 1


def vectorize(
    documents: Sequence[str],
    *,
    kind: str = "tfidf",
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
) -> tuple[Any, TfidfVectorizer | CountVectorizer]:
    """Vectorize preprocessed documents.

    The documents arrive already tokenized and space-joined, so the vectorizer
    must not tokenize again: ``token_pattern=r"\\S+"`` keeps ``c++`` and
    ``scikit-learn`` in one piece, and ``lowercase=False`` avoids a redundant
    pass.

    Args:
        documents: Space-joined token strings from
            :func:`text_preprocessing.join_tokens`.
        kind: ``"tfidf"`` or ``"count"``.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency, as a share of the corpus.
        max_features: Vocabulary cap.
        ngram_range: N-gram sizes to extract.

    Returns:
        A ``(document_term_matrix, fitted_vectorizer)`` pair.

    Raises:
        DatasetError: If the vocabulary comes out empty.
        ValueError: If ``kind`` is unknown.
    """
    if kind not in {"tfidf", "count"}:
        raise ValueError(f"unknown vectorizer kind: {kind}")

    factory = TfidfVectorizer if kind == "tfidf" else CountVectorizer
    vectorizer = factory(
        token_pattern=r"\S+",
        lowercase=False,
        ngram_range=ngram_range,
        min_df=effective_min_df(len(documents), min_df),
        max_df=max_df,
        max_features=max_features,
    )

    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError as exc:  # sklearn raises when every term is pruned
        raise DatasetError(f"vectorization produced an empty vocabulary: {exc}") from exc

    LOGGER.info(
        "Vectorized %d documents into %d features (%s)",
        matrix.shape[0],
        matrix.shape[1],
        kind,
    )
    return matrix, vectorizer


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def fit_lda(
    matrix,
    n_topics: int,
    *,
    random_state: int = RANDOM_STATE,
    max_iter: int = 25,
) -> LatentDirichletAllocation:
    """Fit LDA on a document-term matrix.

    ``learning_method="batch"`` (rather than the online variant) is what makes
    the result reproducible on a corpus this small -- there is no minibatch
    ordering to depend on.

    Args:
        matrix: Document-term matrix.
        n_topics: Number of topics to fit.
        random_state: Seed for the variational initialization.
        max_iter: Maximum number of EM passes.

    Returns:
        The fitted model.
    """
    model = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(matrix)
    return model


def topic_term_distribution(model: LatentDirichletAllocation) -> np.ndarray:
    """Normalize ``components_`` into per-topic term probabilities.

    Args:
        model: A fitted LDA model.

    Returns:
        An array of shape ``(n_topics, n_features)`` whose rows sum to 1.
    """
    components = np.asarray(model.components_, dtype=float)
    return components / components.sum(axis=1, keepdims=True)


def top_terms(
    model: LatentDirichletAllocation,
    feature_names: Sequence[str],
    top_n: int = TOP_KEYWORDS,
) -> list[list[tuple[str, float]]]:
    """Extract the highest-probability terms of every topic.

    Args:
        model: A fitted LDA model.
        feature_names: Vocabulary in feature-column order.
        top_n: Number of terms per topic.

    Returns:
        Per topic, ``(term, weight)`` pairs sorted by descending weight.
    """
    distribution = topic_term_distribution(model)
    keywords: list[list[tuple[str, float]]] = []
    for row in distribution:
        order = np.argsort(row)[::-1][:top_n]
        keywords.append([(str(feature_names[index]), float(row[index])) for index in order])
    return keywords


def document_topic_distribution(model: LatentDirichletAllocation, matrix) -> np.ndarray:
    """Compute per-document topic probabilities.

    Args:
        model: A fitted LDA model.
        matrix: The same document-term matrix the model was fitted on.

    Returns:
        An array of shape ``(n_documents, n_topics)`` whose rows sum to 1.
    """
    distribution = model.transform(matrix)
    totals = distribution.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return distribution / totals


def umass_coherence(
    topic_keywords: Sequence[Sequence[tuple[str, float]]],
    matrix,
    vocabulary: dict[str, int],
) -> float:
    """Score topic interpretability with the UMass coherence measure.

    UMass asks, for every ordered pair of a topic's top terms, how often the
    lower-ranked term appears in documents that contain the higher-ranked one::

        C = mean over pairs (i > j) of log( (D(w_i, w_j) + 1) / D(w_j) )

    It is intrinsic (no external reference corpus needed) and closer to 0 is
    better.  This mirrors gensim's ``u_mass`` so the numbers are comparable
    with the literature.

    Args:
        topic_keywords: Per-topic ``(term, weight)`` pairs, best first.
        matrix: The document-term matrix; only its sparsity pattern is used.
        vocabulary: Term to feature-column mapping.

    Returns:
        Mean coherence across topics, or ``float("nan")`` if no pair could be
        scored.
    """
    binary = (matrix > 0).astype(np.float64)
    scores: list[float] = []

    for keywords in topic_keywords:
        columns = [vocabulary[term] for term, _ in keywords if term in vocabulary]
        if len(columns) < 2:
            continue

        # (n_documents, n_terms) presence block -> pairwise co-document counts.
        selected = binary[:, columns]
        block = np.asarray(
            selected.toarray() if hasattr(selected, "toarray") else selected, dtype=float
        )
        co_document = block.T @ block
        document_frequency = np.diag(co_document)

        pair_scores = [
            np.log((co_document[i, j] + 1.0) / document_frequency[j])
            for i in range(1, len(columns))
            for j in range(i)
            if document_frequency[j] > 0
        ]
        if pair_scores:
            scores.append(float(np.mean(pair_scores)))

    return float(np.mean(scores)) if scores else float("nan")


@dataclass
class TopicScore:
    """One row of the topic-count search.

    Attributes:
        n_topics: Number of topics fitted.
        coherence_umass: UMass coherence (higher, i.e. closer to 0, is better).
        perplexity: Held-in perplexity reported by scikit-learn.
    """

    n_topics: int
    coherence_umass: float
    perplexity: float


def topic_range(start: int, stop: int, step: int) -> list[int]:
    """Build the list of candidate topic counts.

    Args:
        start: Smallest candidate (inclusive).
        stop: Largest candidate (inclusive).
        step: Increment.

    Returns:
        Candidate topic counts, at least ``[start]``.
    """
    candidates = list(range(start, stop + 1, step))
    return candidates or [start]


def select_n_topics(
    matrix,
    feature_names: Sequence[str],
    vocabulary: dict[str, int],
    candidates: Sequence[int],
    *,
    random_state: int = RANDOM_STATE,
    max_iter: int = 25,
) -> tuple[int, LatentDirichletAllocation, list[TopicScore]]:
    """Fit LDA for several topic counts and keep the most coherent one.

    Args:
        matrix: Document-term matrix.
        feature_names: Vocabulary in feature-column order.
        vocabulary: Term to feature-column mapping.
        candidates: Topic counts to try.
        random_state: Seed passed to every fit.
        max_iter: EM passes per fit.

    Returns:
        A ``(best_n_topics, best_model, scores)`` triple; ``scores`` keeps the
        whole search so the report can show why ``k`` was chosen.

    Raises:
        DatasetError: If no candidate could be fitted.
    """
    scores: list[TopicScore] = []
    models: dict[int, LatentDirichletAllocation] = {}

    for n_topics in candidates:
        if n_topics < 2 or n_topics > matrix.shape[0]:
            LOGGER.warning("Skipping k=%d: not compatible with %d documents", n_topics, matrix.shape[0])
            continue
        model = fit_lda(matrix, n_topics, random_state=random_state, max_iter=max_iter)
        keywords = top_terms(model, feature_names)
        score = TopicScore(
            n_topics=n_topics,
            coherence_umass=umass_coherence(keywords, matrix, vocabulary),
            perplexity=float(model.perplexity(matrix)),
        )
        scores.append(score)
        models[n_topics] = model
        LOGGER.info(
            "k=%2d  coherence=%7.3f  perplexity=%10.1f",
            score.n_topics,
            score.coherence_umass,
            score.perplexity,
        )

    if not scores:
        raise DatasetError("no candidate topic count could be fitted")

    usable = [score for score in scores if not np.isnan(score.coherence_umass)]
    best = max(usable or scores, key=lambda score: score.coherence_umass if usable else -score.perplexity)
    LOGGER.info("Selected k=%d", best.n_topics)
    return best.n_topics, models[best.n_topics], scores


# --------------------------------------------------------------------------- #
# Assignment and tag cross-check
# --------------------------------------------------------------------------- #


def assign_dominant_topics(document_topics: np.ndarray) -> pd.DataFrame:
    """Turn a document-topic matrix into per-article assignments.

    The runner-up topic is kept as well: an article that is 0.45 "SQL" and 0.40
    "data engineering" is exactly the kind of link the graph stage needs, and
    that information is lost if only the argmax is stored.

    Args:
        document_topics: Array of shape ``(n_documents, n_topics)``.

    Returns:
        A frame with ``topic_id``, ``topic_prob``, ``topic_second_id`` and
        ``topic_second_prob``, indexed like the input rows.
    """
    order = np.argsort(document_topics, axis=1)[:, ::-1]
    dominant = order[:, 0]
    rows = np.arange(document_topics.shape[0])

    has_runner_up = document_topics.shape[1] > 1
    second = order[:, 1] if has_runner_up else dominant

    return pd.DataFrame(
        {
            "topic_id": dominant.astype("int64"),
            "topic_prob": document_topics[rows, dominant].round(4),
            "topic_second_id": second.astype("int64"),
            "topic_second_prob": (
                document_topics[rows, second].round(4) if has_runner_up else np.zeros(len(rows))
            ),
        }
    )


def tag_profile(
    frame: pd.DataFrame,
    topic_ids: pd.Series,
    n_topics: int,
    top_n: int = TOP_TAGS_PER_TOPIC,
) -> list[list[dict[str, float | str | int]]]:
    """Describe each topic through the tags of the articles assigned to it.

    This is the qualitative cross-check against the existing taxonomy.  Besides
    the raw count, each tag gets a **lift** -- how much more common the tag is
    inside the topic than in the corpus overall.  A tag that is frequent
    everywhere (``Artificial Intelligence``) has a lift near 1 and says nothing;
    a lift of 5 means the topic really did capture that part of the taxonomy.

    Args:
        frame: The cleaned dataset (needs a ``tags`` column).
        topic_ids: Dominant topic per article, aligned with ``frame``.
        n_topics: Number of topics.
        top_n: Tags reported per topic.

    Returns:
        Per topic, a list of ``{tag, count, share, lift}`` dictionaries sorted
        by descending lift among the topic's most frequent tags.
    """
    tags_list = split_tags(frame["tags"]).reset_index(drop=True)
    topics = pd.Series(topic_ids).reset_index(drop=True)

    exploded = pd.DataFrame({"topic_id": topics, "tag": tags_list}).explode("tag").dropna(subset=["tag"])
    corpus_share = exploded["tag"].value_counts(normalize=True) if not exploded.empty else pd.Series(dtype=float)

    profile: list[list[dict[str, float | str | int]]] = []
    for topic_id in range(n_topics):
        subset = exploded[exploded["topic_id"] == topic_id]
        if subset.empty:
            profile.append([])
            continue

        counts = subset["tag"].value_counts()
        total = int(counts.sum())
        entries = [
            {
                "tag": str(tag),
                "count": int(count),
                "share": round(float(count) / total, 4),
                "lift": round(float((count / total) / corpus_share.get(tag, 1e-9)), 2),
            }
            # Rank by count first so rare-tag noise cannot win on lift alone.
            for tag, count in counts.head(top_n * 2).items()
        ]
        entries.sort(key=lambda entry: (-float(entry["lift"]), -int(entry["count"])))
        profile.append(entries[:top_n])

    return profile


def topic_similarity(model: LatentDirichletAllocation) -> np.ndarray:
    """Cosine similarity between topics over their term distributions.

    Args:
        model: A fitted LDA model.

    Returns:
        A symmetric ``(n_topics, n_topics)`` matrix with ones on the diagonal.
    """
    distribution = topic_term_distribution(model)
    norms = np.linalg.norm(distribution, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = distribution / norms
    return normalized @ normalized.T


def similarity_edges(similarity: np.ndarray, threshold: float = EDGE_THRESHOLD) -> list[dict[str, float | int]]:
    """Turn a similarity matrix into a weighted edge list for networkx.

    Args:
        similarity: Output of :func:`topic_similarity`.
        threshold: Minimum weight for an edge to be kept.

    Returns:
        ``{source, target, weight}`` dictionaries, strongest edge first.
    """
    edges = [
        {"source": int(i), "target": int(j), "weight": round(float(similarity[i, j]), 4)}
        for i in range(similarity.shape[0])
        for j in range(i + 1, similarity.shape[0])
        if similarity[i, j] >= threshold
    ]
    edges.sort(key=lambda edge: -float(edge["weight"]))
    return edges


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #


def build_payload(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    keywords: Sequence[Sequence[tuple[str, float]]],
    labels: Sequence[str],
    tags: Sequence[Sequence[dict[str, float | str | int]]],
    similarity: np.ndarray,
    scores: Sequence[TopicScore],
    settings: dict[str, object],
) -> dict[str, object]:
    """Assemble the ``topics.json`` payload.

    Args:
        frame: The cleaned dataset.
        assignments: Output of :func:`assign_dominant_topics`.
        keywords: Per-topic ``(term, weight)`` pairs.
        labels: Per-topic human-readable names.
        tags: Per-topic tag profile.
        similarity: Topic-topic cosine similarity.
        scores: The topic-count search results.
        settings: Model and vectorizer settings, echoed for reproducibility.

    Returns:
        A JSON-serializable dictionary.
    """
    counts = assignments["topic_id"].value_counts()
    total = len(assignments)

    topics = []
    for topic_id, (topic_keywords, label) in enumerate(zip(keywords, labels)):
        members = frame.loc[assignments["topic_id"].to_numpy() == topic_id]
        examples = (
            members.assign(prob=assignments.loc[members.index, "topic_prob"])
            .sort_values("prob", ascending=False)
            .head(EXAMPLES_PER_TOPIC)
        )
        topics.append(
            {
                "topic_id": topic_id,
                "label": label,
                "keywords": [{"term": term, "weight": round(weight, 5)} for term, weight in topic_keywords],
                "article_count": int(counts.get(topic_id, 0)),
                "share": round(float(counts.get(topic_id, 0)) / total, 4) if total else 0.0,
                "top_tags": list(tags[topic_id]),
                "examples": [
                    {"title": str(row.title), "url": str(row.url)} for row in examples.itertuples()
                ],
            }
        )

    return {
        "source": str(settings.get("input", DEFAULT_INPUT)),
        "n_documents": total,
        "n_topics": len(topics),
        "settings": settings,
        "selection": [asdict(score) for score in scores],
        "topics": topics,
        "similarity": {
            "metric": "cosine over topic-term distributions",
            "matrix": [[round(float(value), 4) for value in row] for row in similarity],
        },
        "edges": similarity_edges(similarity),
    }


def write_topics_json(payload: dict[str, object], path: Path) -> None:
    """Persist the topic payload.

    Args:
        payload: Output of :func:`build_payload`.
        path: Destination JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote %d topics to %s", len(payload["topics"]), path)  # type: ignore[arg-type]


def write_articles_csv(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    labels: Sequence[str],
    keywords: Sequence[Sequence[tuple[str, float]]],
    path: Path,
) -> pd.DataFrame:
    """Write the cleaned articles with their dominant topic attached.

    Args:
        frame: The cleaned dataset.
        assignments: Output of :func:`assign_dominant_topics`.
        labels: Per-topic human-readable names.
        keywords: Per-topic ``(term, weight)`` pairs.
        path: Destination CSV path.

    Returns:
        The frame that was written.
    """
    top_keywords = ["|".join(term for term, _ in topic[:5]) for topic in keywords]

    enriched = frame.drop(columns=[c for c in ("tags_list", "published_at") if c in frame.columns]).copy()
    enriched = enriched.reset_index(drop=True)
    for column in assignments.columns:
        enriched[column] = assignments[column].to_numpy()
    enriched["topic_label"] = enriched["topic_id"].map(dict(enumerate(labels)))
    enriched["topic_keywords"] = enriched["topic_id"].map(dict(enumerate(top_keywords)))

    path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(path, index=False, encoding="utf-8")
    LOGGER.info("Wrote %d articles to %s", len(enriched), path)
    return enriched


def format_report(payload: dict[str, object]) -> str:
    """Render the Markdown topic report.

    Args:
        payload: Output of :func:`build_payload`.

    Returns:
        The report text.
    """
    settings = payload["settings"]  # type: ignore[index]
    topics = payload["topics"]  # type: ignore[index]
    scores = payload["selection"]  # type: ignore[index]

    lines = [
        "# Тематичне моделювання: TF-IDF + LDA",
        "",
        "Звіт згенеровано скриптом `analysis/topic_modeling.py` "
        "(перегенерувати: `python analysis/topic_modeling.py`).",
        "",
        "## Параметри запуску",
        "",
        f"- Корпус: `{payload['source']}` — {payload['n_documents']} статей "
        "(текст = `title` + `excerpt`)",
        f"- Векторизація: {settings['vectorizer']}, n-грами {settings['ngram_range']}, "  # type: ignore[index]
        f"min_df={settings['min_df']}, max_df={settings['max_df']}, "  # type: ignore[index]
        f"словник {settings['n_features']} термів",  # type: ignore[index]
        f"- Модель: scikit-learn `LatentDirichletAllocation`, "
        f"`learning_method=batch`, `random_state={settings['random_state']}`",  # type: ignore[index]
        f"- Обрана кількість тем: **{payload['n_topics']}**",
        "",
        "## Підбір кількості тем",
        "",
        "Метрика — UMass coherence (внутрішня когерентність теми: наскільки часто "
        "ключові слова теми зустрічаються разом в одних статтях). Чим ближче до 0, "
        "тим краще. Perplexity наведено довідково.",
        "",
        "| k | Coherence (UMass) | Perplexity |",
        "| --- | --- | --- |",
    ]

    best_k = payload["n_topics"]
    for score in scores:  # type: ignore[union-attr]
        marker = " ✅" if score["n_topics"] == best_k else ""
        lines.append(
            f"| {score['n_topics']}{marker} | {score['coherence_umass']:.3f} | {score['perplexity']:.1f} |"
        )

    lines += ["", "## Теми", ""]

    for topic in topics:  # type: ignore[union-attr]
        keywords = ", ".join(f"`{item['term']}`" for item in topic["keywords"])
        lines += [
            f"### Тема {topic['topic_id']}. {topic['label']}",
            "",
            f"- Статей: **{topic['article_count']}** ({topic['share'] * 100:.1f}% корпусу)",
            f"- Ключові слова: {keywords}",
        ]

        if topic["top_tags"]:
            tags = ", ".join(
                f"{item['tag']} ({item['count']} ст., lift {item['lift']})" for item in topic["top_tags"]
            )
            lines.append(f"- Теги статей теми: {tags}")

        if topic["examples"]:
            lines.append("- Приклади статей:")
            lines += [f"  - [{example['title']}]({example['url']})" for example in topic["examples"]]
        lines.append("")

    lines += [
        "## Звірка з таксономією тегів",
        "",
        "Теги (`tags` в `articles_clean.csv`) — це людська таксономія DataCamp, "
        "тож вони працюють як приблизна «земля правди» для тем, знайдених LDA. "
        "Для кожного тега в темі рахується **lift** — у скільки разів тег частіший "
        "усередині теми, ніж у корпусі загалом:",
        "",
        "- `lift ≈ 1` — тег однаково поширений усюди (напр. наскрізний "
        "`Artificial Intelligence`) і нічого не підтверджує;",
        "- `lift > 2` — тема справді «зловила» окрему частину таксономії.",
        "",
        "Нижче — теми, відсортовані за найсильнішим lift їхнього провідного тега.",
        "",
        "| Тема | Провідний тег | Lift | Частка статей теми з тегом |",
        "| --- | --- | --- | --- |",
    ]

    leading = [
        (topic, topic["top_tags"][0])
        for topic in topics  # type: ignore[union-attr]
        if topic["top_tags"]
    ]
    leading.sort(key=lambda pair: -float(pair[1]["lift"]))
    for topic, tag in leading:
        lines.append(
            f"| {topic['topic_id']}. {topic['label']} | {tag['tag']} | "
            f"{tag['lift']} | {tag['share'] * 100:.0f}% |"
        )

    lines += [
        "",
        "## Наступний крок",
        "",
        "`data/processed/topics.json` уже містить усе, що потрібно для графа "
        "(`viz/`): `topics[].topic_id` — вузли, `edges` — зважені ребра за "
        "косинусною подібністю розподілів слів. Альтернативний спосіб побудувати "
        "ребра — колонки `topic_id` / `topic_second_id` у "
        "`data/processed/articles_with_topics.csv`: скільки статей поєднують дві теми.",
        "",
    ]
    return "\n".join(lines)


def write_report(payload: dict[str, object], path: Path) -> None:
    """Persist the Markdown report.

    Args:
        payload: Output of :func:`build_payload`.
        path: Destination Markdown path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(payload), encoding="utf-8")
    LOGGER.info("Wrote report to %s", path)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_outputs(payload: dict[str, object], articles: pd.DataFrame, n_documents: int) -> None:
    """Assert the invariants the visualization stage relies on.

    Args:
        payload: Output of :func:`build_payload`.
        articles: The enriched article frame.
        n_documents: Expected number of articles.

    Raises:
        AssertionError: If any invariant is violated.
    """
    topics = payload["topics"]  # type: ignore[index]
    n_topics = len(topics)  # type: ignore[arg-type]

    assert n_topics >= 2, "LDA returned fewer than two topics"
    assert [topic["topic_id"] for topic in topics] == list(range(n_topics)), "topic_id must be 0..n-1"  # type: ignore[union-attr]
    assert len({topic["label"] for topic in topics}) == n_topics, "topic labels are not unique"  # type: ignore[union-attr]
    assert all(topic["keywords"] for topic in topics), "a topic has no keywords"  # type: ignore[union-attr]

    assert len(articles) == n_documents, "article count changed during topic assignment"
    assert articles["topic_id"].notna().all(), "unassigned articles"
    assert articles["topic_id"].between(0, n_topics - 1).all(), "topic_id out of range"
    assert articles["topic_prob"].between(0.0, 1.0).all(), "topic_prob out of range"
    assert (
        int(sum(topic["article_count"] for topic in topics)) == n_documents  # type: ignore[union-attr]
    ), "topic article counts do not sum to the corpus size"


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run(
    frame: pd.DataFrame,
    *,
    candidates: Sequence[int],
    vectorizer_kind: str = "tfidf",
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
    max_features: int = DEFAULT_MAX_FEATURES,
    random_state: int = RANDOM_STATE,
    max_iter: int = 25,
    source: str = str(DEFAULT_INPUT),
) -> tuple[dict[str, object], pd.DataFrame]:
    """Run the whole pipeline in memory, without touching the filesystem.

    Keeping I/O out of here is what makes the pipeline testable on a tiny
    in-memory corpus.

    Args:
        frame: The cleaned dataset.
        candidates: Topic counts to try.
        vectorizer_kind: ``"tfidf"`` or ``"count"``.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency.
        max_features: Vocabulary cap.
        random_state: Seed for LDA.
        max_iter: EM passes per fit.
        source: Value recorded as ``source`` in the payload.

    Returns:
        A ``(payload, enriched_frame)`` pair.  ``enriched_frame`` is the article
        table plus topic columns, not yet written to disk.

    Raises:
        DatasetError: If the corpus is unusable.
    """
    documents = build_corpus(frame)
    if not documents:
        raise DatasetError("the input dataset is empty")

    tokens = preprocess_documents(documents)
    matrix, vectorizer = vectorize(
        join_tokens(tokens),
        kind=vectorizer_kind,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )

    feature_names = list(vectorizer.get_feature_names_out())
    vocabulary = {term: index for index, term in enumerate(feature_names)}

    n_topics, model, scores = select_n_topics(
        matrix,
        feature_names,
        vocabulary,
        candidates,
        random_state=random_state,
        max_iter=max_iter,
    )

    keywords = top_terms(model, feature_names)
    labels = assign_labels(keywords)
    assignments = assign_dominant_topics(document_topic_distribution(model, matrix))
    tags = tag_profile(frame, assignments["topic_id"], n_topics)
    similarity = topic_similarity(model)

    settings: dict[str, object] = {
        "input": source,
        "library": "scikit-learn",
        "algorithm": "LatentDirichletAllocation",
        "vectorizer": vectorizer_kind,
        "ngram_range": list(NGRAM_RANGE),
        "min_df": effective_min_df(len(documents), min_df),
        "max_df": max_df,
        "max_features": max_features,
        "n_features": int(matrix.shape[1]),
        "random_state": random_state,
        "max_iter": max_iter,
        "candidates": list(candidates),
    }

    payload = build_payload(
        frame.reset_index(drop=True),
        assignments,
        keywords,
        labels,
        tags,
        similarity,
        scores,
        settings,
    )
    return payload, assignments


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="cleaned CSV path")
    parser.add_argument("--topics-output", type=Path, default=DEFAULT_TOPICS_JSON, help="topics JSON path")
    parser.add_argument(
        "--articles-output", type=Path, default=DEFAULT_ARTICLES_CSV, help="articles + topic CSV path"
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT, help="Markdown report path")
    parser.add_argument(
        "--topics", type=int, default=None, help="fix the number of topics and skip the search"
    )
    parser.add_argument(
        "--topic-range",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        default=DEFAULT_TOPIC_RANGE,
        help="candidate topic counts to evaluate",
    )
    parser.add_argument(
        "--vectorizer", choices=("tfidf", "count"), default="tfidf", help="document-term weighting"
    )
    parser.add_argument("--min-df", type=int, default=DEFAULT_MIN_DF, help="minimum document frequency")
    parser.add_argument("--max-df", type=float, default=DEFAULT_MAX_DF, help="maximum document frequency")
    parser.add_argument(
        "--max-features", type=int, default=DEFAULT_MAX_FEATURES, help="vocabulary size cap"
    )
    parser.add_argument("--max-iter", type=int, default=25, help="LDA EM iterations")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="random seed")
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

    if not args.input.exists():
        LOGGER.error("%s not found - run analysis/clean_data.py first", args.input)
        return 2

    frame = load_clean(args.input)
    candidates = [args.topics] if args.topics else topic_range(*args.topic_range)

    try:
        payload, assignments = run(
            frame,
            candidates=candidates,
            vectorizer_kind=args.vectorizer,
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            random_state=args.random_state,
            max_iter=args.max_iter,
            source=str(args.input),
        )
    except DatasetError as exc:
        LOGGER.error("%s", exc)
        return 2

    labels = [str(topic["label"]) for topic in payload["topics"]]  # type: ignore[union-attr,index]
    keywords = [
        [(str(item["term"]), float(item["weight"])) for item in topic["keywords"]]
        for topic in payload["topics"]  # type: ignore[union-attr,index]
    ]

    articles = write_articles_csv(
        frame.reset_index(drop=True), assignments, labels, keywords, args.articles_output
    )
    validate_outputs(payload, articles, len(frame))

    write_topics_json(payload, args.topics_output)
    write_report(payload, args.report_output)

    print(f"{payload['n_topics']} topics over {payload['n_documents']} articles")
    for topic in payload["topics"]:  # type: ignore[union-attr]
        print(f"  {topic['topic_id']:>2}  {topic['article_count']:>4} articles  {topic['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
