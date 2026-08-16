#!/usr/bin/env python3
"""Offline checks for analysis/text_preprocessing.py and analysis/topic_modeling.py.

Everything runs on a small synthetic corpus built in this file -- the real
1478-article dataset is never needed, so the suite passes in a fresh checkout
and needs no network access.

The fake corpus has three deliberately separable themes (SQL, deep learning,
careers).  That is enough to assert the mechanics of the pipeline; it is *not*
enough to assert that LDA recovers exactly those three themes, so no test makes
that claim.

Run with:
    python tests/test_topic_modeling.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import text_preprocessing as tp  # noqa: E402
import topic_modeling as tm  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

#: ``(title, excerpt, tags)`` triples, six per theme.
FAKE_ARTICLES: list[tuple[str, str, str]] = [
    # -- SQL / databases ---------------------------------------------------- #
    ("SQL Joins Explained", "Learn how to join tables in SQL and write faster queries.", "SQL"),
    ("An Introduction to SQL Queries", "Writing SELECT queries against a relational database.", "SQL"),
    ("PostgreSQL vs MySQL", "Comparing two relational database engines for analytics teams.", "SQL|Data Engineering"),
    ("Window Functions in SQL", "Ranking rows inside a database query without subqueries.", "SQL"),
    ("Database Normalization", "Designing a relational schema with normalized tables.", "SQL"),
    ("NoSQL Databases Explained", "When a document database beats a relational database.", "SQL|Data Engineering"),
    # -- Deep learning ------------------------------------------------------ #
    ("Deep Learning with PyTorch", "Training a neural network on image data with PyTorch.", "Deep Learning"),
    ("Understanding Transformers", "How the transformer neural network changed deep learning.", "Deep Learning"),
    ("Convolutional Neural Networks", "Training a CNN for computer vision with TensorFlow.", "Deep Learning"),
    ("Introduction to LLMs", "Large language models, prompts and the transformer stack.", "Large Language Models"),
    ("Fine-Tuning an LLM", "Adapting a pretrained language model with your own data.", "Large Language Models"),
    ("Neural Network Basics", "Backpropagation, gradients and training a neural network.", "Deep Learning"),
    # -- Careers ------------------------------------------------------------ #
    ("Data Scientist Career Guide", "Skills, salary and interview questions for a data scientist job.", "Careers"),
    ("How to Become a Data Analyst", "A career roadmap and the skills a data analyst job needs.", "Careers"),
    ("Data Science Interview Questions", "Common interview questions asked when hiring a data scientist.", "Careers"),
    ("Building a Data Portfolio", "Portfolio projects that get a beginner a first data job.", "Careers"),
    ("Data Analyst Salary Guide", "What a data analyst salary looks like across the job market.", "Careers"),
    ("Certification vs Degree", "Whether a certification beats a degree for a data career.", "Careers"),
]


def fake_frame() -> pd.DataFrame:
    """Build a cleaned-dataset-shaped frame from :data:`FAKE_ARTICLES`.

    Returns:
        A frame with the schema ``clean_data.py`` produces.
    """
    rows = []
    for index, (title, excerpt, tags) in enumerate(FAKE_ARTICLES):
        rows.append(
            {
                "title": title,
                "url": f"https://x.test/blog/post-{index}",
                "published_date": f"2026-01-{index % 28 + 1:02d}",
                "author": "Tester",
                "tags": tags,
                "excerpt": excerpt,
                "scraped_at": "2026-08-16",
            }
        )
    return pd.DataFrame(rows)


def write_clean_csv(directory: Path) -> Path:
    """Write the fake corpus as an ``articles_clean.csv`` file.

    Args:
        directory: Temporary directory.

    Returns:
        Path to the written CSV.
    """
    path = directory / "articles_clean.csv"
    fake_frame().to_csv(path, index=False, encoding="utf-8")
    return path


def fitted_pipeline(n_topics: int = 3):
    """Vectorize and fit the fake corpus once.

    Args:
        n_topics: Number of topics to fit.

    Returns:
        A ``(matrix, vectorizer, model, feature_names)`` tuple.
    """
    frame = fake_frame()
    tokens = tp.preprocess_documents(tm.build_corpus(frame))
    matrix, vectorizer = tm.vectorize(tp.join_tokens(tokens), min_df=1, max_df=1.0)
    model = tm.fit_lda(matrix, n_topics, max_iter=10)
    return matrix, vectorizer, model, list(vectorizer.get_feature_names_out())


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #


def test_tokenize_keeps_domain_notation() -> None:
    """``c++``, ``scikit-learn`` and ``a/b`` survive tokenization intact."""
    tokens = tp.tokenize("Using C++ with scikit-learn for A/B tests")
    assert "c++" in tokens, tokens
    assert "scikit-learn" in tokens, tokens
    assert "a/b" in tokens, tokens


def test_tokenize_drops_numeric_tokens() -> None:
    """Years and version numbers carry no topical signal."""
    tokens = tp.tokenize("Top 10 AI trends of 2026")
    assert "2026" not in tokens
    assert "10" not in tokens
    assert "ai" in tokens


def test_tokenize_folds_variants() -> None:
    """Spelling variants collapse onto one canonical form."""
    assert "llm" in tp.tokenize("A guide to LLMs")
    assert "llms" not in tp.tokenize("A guide to LLMs")
    assert tp.tokenize("machine-learning") == ["machine", "learning"]


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_stopwords_are_removed() -> None:
    """Function words never reach the vectorizer."""
    tokens = tp.preprocess_documents(["The best of the data with a lot of the things"])[0]
    assert "the" not in tokens
    assert "with" not in tokens
    assert "data" in tokens


def test_plurals_are_lemmatized() -> None:
    """Safe plural rules apply without a corpus lookup."""
    tokens = tp.preprocess_documents(["Models, queries and databases"])[0]
    assert "model" in tokens, tokens
    assert "query" in tokens, tokens
    assert "database" in tokens, tokens


def test_domain_terms_are_not_mangled() -> None:
    """Short and domain-specific terms survive both filters and lemmatization."""
    tokens = tp.preprocess_documents(["SQL, AI and R for analytics and statistics"])[0]
    for term in ("sql", "ai", "r", "analytics", "statistics"):
        assert term in tokens, (term, tokens)


def test_verb_suffix_stripped_only_when_stem_is_attested() -> None:
    """``coding`` -> ``code`` because the corpus uses ``code``; ``writing`` stays."""
    documents = ["Clean code matters", "Coding interviews and writing style"]
    tokens = tp.preprocess_documents(documents)
    assert "code" in tokens[1], tokens
    assert "cod" not in tokens[1], tokens
    # `write` never occurs in this corpus, so nothing licenses `writ`.
    assert "writing" in tokens[1], tokens


def test_preprocessing_is_deterministic() -> None:
    """The same corpus always produces the same tokens."""
    documents = [title for title, _, _ in FAKE_ARTICLES]
    assert tp.preprocess_documents(documents) == tp.preprocess_documents(documents)


# --------------------------------------------------------------------------- #
# Corpus and vectorization
# --------------------------------------------------------------------------- #


def test_build_corpus_joins_title_and_excerpt() -> None:
    """Each document is ``title. excerpt``."""
    documents = tm.build_corpus(fake_frame())
    assert len(documents) == len(FAKE_ARTICLES)
    assert documents[0].startswith("SQL Joins Explained. ")
    assert documents[0].endswith("faster queries.")


def test_build_corpus_requires_text_columns() -> None:
    """A frame without ``excerpt`` is rejected with a clear error."""
    try:
        tm.build_corpus(pd.DataFrame({"title": ["x"]}))
    except tm.DatasetError as exc:
        assert "excerpt" in str(exc)
    else:
        raise AssertionError("expected DatasetError")


def test_effective_min_df_relaxes_on_small_corpora() -> None:
    """A 20-document corpus would be emptied by the production ``min_df``."""
    assert tm.effective_min_df(20, 5) == 1
    assert tm.effective_min_df(1478, 5) == 5


def test_vectorize_returns_matrix_and_vocabulary() -> None:
    """The matrix has one row per document and a non-empty vocabulary."""
    matrix, vectorizer, _, features = fitted_pipeline()
    assert matrix.shape[0] == len(FAKE_ARTICLES)
    assert matrix.shape[1] == len(features) > 0
    assert "sql" in features
    # Bigrams are enabled, so multi-word concepts exist as single features.
    assert any(" " in feature for feature in features)


def test_vectorize_rejects_unknown_kind() -> None:
    """Only ``tfidf`` and ``count`` are supported."""
    try:
        tm.vectorize(["sql query"], kind="hashing")
    except ValueError as exc:
        assert "hashing" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def test_fit_lda_returns_requested_number_of_topics() -> None:
    """The model exposes exactly ``n_topics`` term distributions."""
    for n_topics in (2, 3, 5):
        _, _, model, _ = fitted_pipeline(n_topics)
        assert model.components_.shape[0] == n_topics


def test_topic_term_distribution_rows_are_probabilities() -> None:
    """Each topic's term weights sum to 1."""
    _, _, model, _ = fitted_pipeline()
    distribution = tm.topic_term_distribution(model)
    assert np.allclose(distribution.sum(axis=1), 1.0)


def test_top_terms_are_sorted_and_sized() -> None:
    """Keywords come back highest-weight first, ``top_n`` of them."""
    _, _, model, features = fitted_pipeline()
    keywords = tm.top_terms(model, features, top_n=5)
    assert len(keywords) == model.components_.shape[0]
    for topic in keywords:
        assert len(topic) == 5
        weights = [weight for _, weight in topic]
        assert weights == sorted(weights, reverse=True)
        assert all(term in features for term, _ in topic)


def test_document_topic_distribution_rows_sum_to_one() -> None:
    """Every article gets a proper probability distribution over topics."""
    matrix, _, model, _ = fitted_pipeline()
    distribution = tm.document_topic_distribution(model, matrix)
    assert distribution.shape == (len(FAKE_ARTICLES), model.components_.shape[0])
    assert np.allclose(distribution.sum(axis=1), 1.0)


def test_umass_coherence_is_a_finite_number() -> None:
    """Coherence is computable on the fake corpus."""
    matrix, _, model, features = fitted_pipeline()
    vocabulary = {term: index for index, term in enumerate(features)}
    score = tm.umass_coherence(tm.top_terms(model, features), matrix, vocabulary)
    assert np.isfinite(score)


def test_umass_coherence_prefers_co_occurring_terms() -> None:
    """A topic whose keywords always co-occur scores above a scattered one."""
    matrix, _, _, features = fitted_pipeline()
    vocabulary = {term: index for index, term in enumerate(features)}
    together = [[("sql", 0.2), ("query", 0.1), ("database", 0.1)]]
    scattered = [[("sql", 0.2), ("career", 0.1), ("neural", 0.1)]]
    assert tm.umass_coherence(together, matrix, vocabulary) > tm.umass_coherence(
        scattered, matrix, vocabulary
    )


def test_topic_range_is_inclusive() -> None:
    """``--topic-range 8 12 2`` means 8, 10 and 12."""
    assert tm.topic_range(8, 12, 2) == [8, 10, 12]
    assert tm.topic_range(4, 4, 2) == [4]


def test_select_n_topics_picks_a_candidate() -> None:
    """The search returns one of the candidates plus a score per candidate."""
    matrix, vectorizer, _, features = fitted_pipeline()
    vocabulary = {term: index for index, term in enumerate(features)}
    best, model, scores = tm.select_n_topics(matrix, features, vocabulary, [2, 3], max_iter=10)
    assert best in {2, 3}
    assert [score.n_topics for score in scores] == [2, 3]
    assert model.components_.shape[0] == best


# --------------------------------------------------------------------------- #
# Assignment, labels, tags
# --------------------------------------------------------------------------- #


def test_assign_dominant_topics_ranks_two_topics() -> None:
    """The dominant topic wins and the runner-up is kept for the graph stage."""
    distribution = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    assignments = tm.assign_dominant_topics(distribution)
    assert list(assignments["topic_id"]) == [0, 2]
    assert list(assignments["topic_second_id"]) == [1, 1]
    assert (assignments["topic_prob"] >= assignments["topic_second_prob"]).all()
    assert assignments["topic_prob"].between(0.0, 1.0).all()


def test_phrase_matching_is_whole_word() -> None:
    """``ai`` must not match ``train``, but must match ``generative ai``."""
    assert tm._phrase_in_term("ai", "generative ai")
    assert not tm._phrase_in_term("ai", "train")
    assert tm._phrase_in_term("machine learning", "machine learning model")
    assert not tm._phrase_in_term("machine learning", "learning machine")


def test_label_topic_uses_the_rule_catalogue() -> None:
    """A SQL-flavoured topic gets the SQL label."""
    label, score = tm.label_topic([("sql", 0.10), ("query", 0.05), ("database", 0.04)])
    assert label == "SQL та бази даних"
    assert score > 0


def test_label_topic_returns_none_when_nothing_matches() -> None:
    """Unmatched topics fall back to a keyword-derived name."""
    label, score = tm.label_topic([("zzz", 0.1), ("qqq", 0.05)])
    assert label is None
    assert score == 0
    assert tm.fallback_label(7, [("zzz", 0.1), ("qqq", 0.05)]).startswith("Тема 7: zzz, qqq")


def test_assign_labels_are_unique() -> None:
    """Two SQL-looking topics never end up with the same name."""
    labels = tm.assign_labels(
        [
            [("sql", 0.20), ("query", 0.10)],
            [("sql", 0.05), ("join", 0.04)],
            [("zzz", 0.03)],
        ]
    )
    assert len(set(labels)) == 3
    assert labels[0] == "SQL та бази даних"
    assert labels[2].startswith("Тема 2:")


def test_tag_profile_reports_lift() -> None:
    """A tag concentrated in one topic gets a lift well above 1."""
    frame = fake_frame()
    # Assign each theme block to its own topic, matching how the corpus is built.
    topic_ids = pd.Series([0] * 6 + [1] * 6 + [2] * 6)
    profile = tm.tag_profile(frame, topic_ids, n_topics=3)
    assert len(profile) == 3
    leading = {entry[0]["tag"] for entry in profile}
    assert {"SQL", "Careers"} <= leading, profile
    for entries in profile:
        assert entries[0]["lift"] > 1.0


def test_topic_similarity_is_symmetric_with_unit_diagonal() -> None:
    """Cosine similarity behaves as expected, so the edge list is well-formed."""
    _, _, model, _ = fitted_pipeline()
    similarity = tm.topic_similarity(model)
    assert np.allclose(np.diag(similarity), 1.0)
    assert np.allclose(similarity, similarity.T)


def test_similarity_edges_respect_the_threshold() -> None:
    """Edges are upper-triangular, filtered and sorted by weight."""
    similarity = np.array([[1.0, 0.5, 0.05], [0.5, 1.0, 0.9], [0.05, 0.9, 1.0]])
    edges = tm.similarity_edges(similarity, threshold=0.1)
    assert [(edge["source"], edge["target"]) for edge in edges] == [(1, 2), (0, 1)]
    assert all(edge["source"] < edge["target"] for edge in edges)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_run_produces_a_complete_payload() -> None:
    """The in-memory pipeline returns a graph-ready payload."""
    frame = fake_frame()
    payload, assignments = tm.run(frame, candidates=[3], max_iter=10, max_df=1.0)

    assert payload["n_topics"] == 3
    assert payload["n_documents"] == len(FAKE_ARTICLES)
    assert [topic["topic_id"] for topic in payload["topics"]] == [0, 1, 2]
    assert all(len(topic["keywords"]) > 0 for topic in payload["topics"])
    assert len(payload["similarity"]["matrix"]) == 3
    assert len(assignments) == len(FAKE_ARTICLES)
    # Every article is accounted for exactly once.
    assert sum(topic["article_count"] for topic in payload["topics"]) == len(FAKE_ARTICLES)


def test_run_is_deterministic() -> None:
    """Two runs on the same input produce identical topics."""
    frame = fake_frame()
    first, _ = tm.run(frame, candidates=[3], max_iter=10, max_df=1.0)
    second, _ = tm.run(frame, candidates=[3], max_iter=10, max_df=1.0)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_main_writes_every_artifact() -> None:
    """The CLI writes the JSON, the CSV and the Markdown report."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        topics_json = directory / "topics.json"
        articles_csv = directory / "articles_with_topics.csv"
        report_md = directory / "report.md"

        exit_code = tm.main(
            [
                "--input", str(write_clean_csv(directory)),
                "--topics-output", str(topics_json),
                "--articles-output", str(articles_csv),
                "--report-output", str(report_md),
                "--topics", "3",
                "--max-df", "1.0",
                "--max-iter", "10",
            ]
        )
        assert exit_code == 0

        payload = json.loads(topics_json.read_text(encoding="utf-8"))
        assert payload["n_topics"] == 3
        assert len(payload["topics"]) == 3

        articles = pd.read_csv(articles_csv)
        assert len(articles) == len(FAKE_ARTICLES)
        for column in ("topic_id", "topic_prob", "topic_label", "topic_second_id", "topic_keywords"):
            assert column in articles.columns, column
        assert articles["topic_id"].between(0, 2).all()
        # The original columns are preserved for downstream joins.
        assert {"title", "url", "tags", "excerpt"} <= set(articles.columns)

        report = report_md.read_text(encoding="utf-8")
        assert "# Тематичне моделювання" in report
        for topic in payload["topics"]:
            assert topic["label"] in report


def test_main_reports_missing_input() -> None:
    """A missing dataset exits with code 2 instead of raising."""
    with tempfile.TemporaryDirectory() as tmp:
        assert tm.main(["--input", str(Path(tmp) / "nope.csv")]) == 2


def test_validate_outputs_catches_a_bad_assignment() -> None:
    """The invariants actually fail when an article points at no topic."""
    frame = fake_frame()
    payload, assignments = tm.run(frame, candidates=[3], max_iter=10, max_df=1.0)
    broken = frame.assign(topic_id=99, topic_prob=1.0)
    try:
        tm.validate_outputs(payload, broken, len(frame))
    except AssertionError:
        pass
    else:
        raise AssertionError("expected the topic_id range check to fail")


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
