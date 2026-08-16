#!/usr/bin/env python3
"""Dependency-free English preprocessing for the DataCamp article corpus.

The corpus is short text -- ``title`` plus ``excerpt``, roughly 30-45 words per
article -- so preprocessing decides how readable the LDA topics end up being.
Three choices drive this module:

*Offline by design.*  ``nltk`` and ``spaCy`` both need a model/corpus download
at first use.  A network call inside an analysis step makes the pipeline
non-reproducible in CI and in a fresh checkout, so tokenizer, stop-word list
and lemmatizer are all implemented here with the standard library.

*Domain terms are not noise.*  ``SQL``, ``AI``, ``R``, ``BI``, ``C++`` would be
destroyed by a naive ``len(token) > 2`` filter or by a stemmer.  They are kept
verbatim through :data:`DOMAIN_TERMS`, and spelling variants are folded
together by :data:`CANONICAL_TERMS` (``llms`` -> ``llm``).

*Lemmatization is corpus-aware.*  Blind suffix stripping turns ``coding`` into
``cod`` and ``writing`` into ``writ``, which wrecks the human-readable topic
labels this whole pipeline exists to produce.  Instead a suffix is only removed
when the resulting stem is itself attested in the corpus (``coding`` -> ``code``
because ``code`` occurs; ``writing`` -> ``write``).  Plurals, which are safe,
are handled by plain rules.

Typical use:

    from text_preprocessing import preprocess_documents

    tokens = preprocess_documents(["Learn SQL joins", "Deep learning models"])
    # [['learn', 'sql', 'join'], ['deep', 'learning', 'model']]
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

#: Standard English function words.  Kept explicit (rather than imported from
#: nltk) so the pipeline runs with no downloads and the list is auditable.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all almost along already also although always am among an and
    another any anyone anything are around as at back be became because become becomes been before
    being below best better between both but by came can cannot could did do does doing done down
    during each either else enough etc even ever every everything few first for from further get gets
    getting give given go goes going good got had has have having he her here hers herself him himself
    his how however i if in into is it its itself just keep kept know known let like likely little
    made make makes making many may maybe me might more moreover most much must my myself need needs
    never new next no nor not nothing now of off often on once one only onto or other others otherwise
    our ours ourselves out over own per perhaps put rather really same say says see seem seen several
    shall she should show shows simply since so some someone something soon still such take taken than
    that the their theirs them themselves then there therefore these they thing things this those
    though three through thus to together too took toward two under until up upon us use used useful
    uses using usually very want was way ways we well went were what when where whether which while who
    whom whose why will with within without would yet you your yours yourself
    """.split()
)

#: Boilerplate that survives cleaning but carries no topical signal.  ``guide``
#: and ``tutorial`` are deliberately *not* here: a "how-to content" topic is a
#: real, interpretable topic in this corpus.
EXTRA_STOPWORDS: frozenset[str] = frozenset(
    """
    datacamp com www http https blog article articles post posts read reading today
    lot lots kind sort thanks welcome
    """.split()
)

#: Terms that must survive verbatim: never lemmatized, never dropped for being
#: short.  Without this, ``r`` and ``ai`` vanish and ``analytics`` becomes
#: ``analytic``.
DOMAIN_TERMS: frozenset[str] = frozenset(
    """
    r sql nosql mysql postgresql postgres sqlite oracle snowflake databricks bigquery redshift
    mongodb cassandra dbt airflow spark hadoop kafka etl elt olap oltp
    ai agi ml dl nlp llm llms rag mlops llmops genai ocr cv rl
    bi kpi roi crm erp saas api apis sdk cli ide gui ui ux
    aws gcp azure gcloud ec2 s3 docker kubernetes k8s terraform git github gitlab
    python r-lang pandas numpy scipy sklearn scikit-learn pytorch tensorflow keras xgboost lightgbm
    matplotlib seaborn plotly streamlit jupyter notebook anaconda conda pip
    java javascript scala julia c++ c# .net php ruby rust golang bash powershell excel
    tableau powerbi looker qlik sheets
    gpt gpt-4 gpt-5 chatgpt claude gemini llama mistral bert transformer transformers
    cnn rnn lstm gan vae svm knn pca tsne umap
    a/b eda roc auc rmse mae mse f1
    learning engineering computing programming analytics modeling modelling marketing forecasting
    clustering embedding embeddings tuning testing mining reasoning
    science scientist statistics analysis business
    """.split()
)

#: Spelling and morphology variants folded onto one canonical surface form.
#: Applied at tokenization time, before lemmatization.
CANONICAL_TERMS: dict[str, str] = {
    "llms": "llm",
    "ais": "ai",
    "apis": "api",
    "kpis": "kpi",
    "embeddings": "embedding",
    "transformers": "transformer",
    "modelling": "modeling",
    "power-bi": "powerbi",
    "powerbi": "powerbi",
    "scikit": "scikit-learn",
    "sklearn": "scikit-learn",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "chat-gpt": "chatgpt",
    "gpt4": "gpt-4",
    "gpt5": "gpt-5",
    "data-science": "data science",
    "machine-learning": "machine learning",
    "deep-learning": "deep learning",
    "a-b": "a/b",
}

#: Irregular forms that no suffix rule can reach.
IRREGULAR_LEMMAS: dict[str, str] = {
    "analyses": "analysis",
    "bases": "basis",
    "children": "child",
    "criteria": "criterion",
    "data": "data",
    "indices": "index",
    "indexes": "index",
    "matrices": "matrix",
    "economics": "economics",
    "mathematics": "mathematics",
    "media": "media",
    "men": "man",
    # `series` would otherwise become `sery`, and `news` would become `new`.
    "news": "news",
    "physics": "physics",
    "series": "series",
    "people": "person",
    "phenomena": "phenomenon",
    "schemas": "schema",
    "schemata": "schema",
    "women": "woman",
    "is": "be",
    "are": "be",
    "was": "be",
    "were": "be",
    "has": "have",
    "had": "have",
}

#: Minimum length for a token that is not a known domain term.
MIN_TOKEN_LENGTH = 3

#: ``c++`` / ``c#`` first, then dotted-hyphenated-slashed words such as
#: ``scikit-learn``, ``gpt-4``, ``a/b``, ``.net``.
TOKEN_RE = re.compile(r"[a-z][a-z0-9]*(?:\+\+|#)|\.?[a-z0-9]+(?:[-._/][a-z0-9]+)*")

#: Tokens made only of digits and punctuation (years, version numbers).
NUMERIC_RE = re.compile(r"^[\d.\-/_]+$")

#: Plural ``-s`` is unsafe after these endings (``analysis``, ``business``,
#: ``bias``, ``chaos``, ``virus``).
UNSAFE_PLURAL_ENDINGS = ("ss", "is", "us", "as", "os", "ys")

#: Suffixes stripped only when the resulting stem is attested in the corpus.
#: ``-er`` is deliberately absent: it would turn ``engineer`` into ``engine``
#: and ``career`` into ``care`` whenever those stems happen to occur, and agent
#: nouns are far more common in this corpus than comparatives.
CORPUS_AWARE_SUFFIXES = ("ing", "ed")


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #


def tokenize(text: str) -> list[str]:
    """Split raw text into lower-case tokens, keeping domain notation intact.

    Args:
        text: Any string; ``None``-ish values are treated as empty.

    Returns:
        Tokens in document order, with numeric-only tokens removed and
        :data:`CANONICAL_TERMS` applied.  Multi-word canonical forms (for
        example ``machine-learning`` -> ``machine learning``) are expanded back
        into separate tokens so that the vectorizer's bigrams still see them.
    """
    if not text:
        return []

    tokens: list[str] = []
    for match in TOKEN_RE.findall(text.lower()):
        # Trailing punctuation is noise; a leading dot is meaningful (`.net`).
        token = match.strip("-_/").rstrip(".")
        if not token or NUMERIC_RE.match(token):
            continue
        canonical = CANONICAL_TERMS.get(token, token)
        tokens.extend(canonical.split(" ") if " " in canonical else [canonical])
    return tokens


def build_vocabulary(token_lists: Iterable[Sequence[str]]) -> Counter[str]:
    """Count raw (pre-lemmatization) tokens across the corpus.

    The counts are what makes lemmatization corpus-aware: a suffix is stripped
    only when the stem it produces is a word the corpus actually uses.

    Args:
        token_lists: Tokenized documents.

    Returns:
        Token frequencies over the whole corpus.
    """
    vocabulary: Counter[str] = Counter()
    for tokens in token_lists:
        vocabulary.update(tokens)
    return vocabulary


# --------------------------------------------------------------------------- #
# Lemmatization
# --------------------------------------------------------------------------- #


class Lemmatizer:
    """Rule-based lemmatizer that checks risky rewrites against the corpus.

    Plural rules are applied unconditionally because they are safe on English
    nouns once the unsafe endings in :data:`UNSAFE_PLURAL_ENDINGS` are excluded.
    Verb suffixes (``-ing``, ``-ed``) are only stripped when the stem -- possibly
    with a restored ``e`` or an undoubled consonant -- occurs in the corpus,
    which is what prevents ``coding`` -> ``cod``.

    Attributes:
        vocabulary: Raw token frequencies used to validate stems.
        protected: Terms returned unchanged.
    """

    def __init__(
        self,
        vocabulary: Counter[str] | None = None,
        protected: Iterable[str] = DOMAIN_TERMS,
    ) -> None:
        """Initialize the lemmatizer.

        Args:
            vocabulary: Raw token counts from :func:`build_vocabulary`.  When
                omitted, corpus-aware suffix stripping is disabled and only the
                safe plural rules apply.
            protected: Terms that must never be rewritten.
        """
        self.vocabulary: Counter[str] = vocabulary if vocabulary is not None else Counter()
        self.protected: frozenset[str] = frozenset(protected)

    def _known(self, candidate: str) -> bool:
        """Check whether a candidate stem is attested in the corpus.

        Args:
            candidate: Proposed stem.

        Returns:
            ``True`` if the corpus contains the stem as a standalone token.
        """
        return len(candidate) >= MIN_TOKEN_LENGTH and candidate in self.vocabulary

    @staticmethod
    def _depluralize(token: str) -> str | None:
        """Apply the safe English plural rules.

        Args:
            token: A lower-case token.

        Returns:
            The singular form, or ``None`` when no rule applies.
        """
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith(("sses", "shes", "ches", "xes", "zes")):
            return token[:-2]
        if (
            token.endswith("s")
            and len(token) > MIN_TOKEN_LENGTH
            and not token.endswith(UNSAFE_PLURAL_ENDINGS)
        ):
            return token[:-1]
        return None

    def _strip_verb_suffix(self, token: str) -> str | None:
        """Strip ``-ing`` / ``-ed`` when the corpus backs the stem.

        Args:
            token: A lower-case token.

        Returns:
            The validated stem, or ``None`` when nothing can be stripped
            safely.
        """
        for suffix in CORPUS_AWARE_SUFFIXES:
            if not token.endswith(suffix) or len(token) - len(suffix) < MIN_TOKEN_LENGTH:
                continue
            stem = token[: -len(suffix)]
            # `coding` -> `code`, `running` -> `run`, `trained` -> `train`.
            candidates = [stem, stem + "e"]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                candidates.append(stem[:-1])
            for candidate in candidates:
                if self._known(candidate):
                    return candidate
        return None

    def lemma(self, token: str) -> str:
        """Reduce one token to its lemma.

        Args:
            token: A lower-case token.

        Returns:
            The lemma, or the token itself when no rule fires.
        """
        if token in self.protected:
            return token
        if token in IRREGULAR_LEMMAS:
            return IRREGULAR_LEMMAS[token]

        singular = self._depluralize(token)
        if singular is not None:
            token = singular
            if token in self.protected:
                return token

        return self._strip_verb_suffix(token) or token

    def __call__(self, token: str) -> str:
        """Alias for :meth:`lemma`.

        Args:
            token: A lower-case token.

        Returns:
            The lemma.
        """
        return self.lemma(token)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def is_content_token(token: str, stopwords: frozenset[str]) -> bool:
    """Decide whether a token carries topical signal.

    Args:
        token: A lower-case token.
        stopwords: Words to reject.

    Returns:
        ``True`` if the token should reach the vectorizer.
    """
    if token in DOMAIN_TERMS:
        return True
    if token in stopwords:
        return False
    return len(token) >= MIN_TOKEN_LENGTH


def default_stopwords() -> frozenset[str]:
    """Build the stop-word set used by the pipeline.

    Returns:
        English function words plus corpus boilerplate, minus anything listed
        as a domain term (``business`` is both a stop-word candidate and a real
        topic word here).
    """
    return frozenset((ENGLISH_STOPWORDS | EXTRA_STOPWORDS) - DOMAIN_TERMS)


def preprocess_documents(
    texts: Sequence[str],
    stopwords: frozenset[str] | None = None,
) -> list[list[str]]:
    """Tokenize, lemmatize and filter a whole corpus.

    The corpus is processed as a batch (not document by document) because the
    lemmatizer needs global token counts before it can rewrite anything.

    Args:
        texts: Raw documents, one string per article.
        stopwords: Override for :func:`default_stopwords`.

    Returns:
        One token list per input document, aligned with ``texts``.
    """
    stops = default_stopwords() if stopwords is None else stopwords
    tokenized = [tokenize(text) for text in texts]
    lemmatizer = Lemmatizer(build_vocabulary(tokenized))

    processed: list[list[str]] = []
    for tokens in tokenized:
        kept: list[str] = []
        for token in tokens:
            # Filter twice: before lemmatization (`using`) and after it
            # (`uses` -> `use`), so both surface forms are caught.
            if token in stops:
                continue
            lemma = lemmatizer.lemma(token)
            if is_content_token(lemma, stops):
                kept.append(lemma)
        processed.append(kept)
    return processed


def join_tokens(token_lists: Iterable[Sequence[str]]) -> list[str]:
    """Flatten token lists back into strings for scikit-learn vectorizers.

    Args:
        token_lists: Preprocessed documents.

    Returns:
        Space-joined documents.
    """
    return [" ".join(tokens) for tokens in token_lists]
