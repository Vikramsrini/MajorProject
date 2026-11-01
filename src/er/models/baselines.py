"""Baseline matchers: string-similarity threshold and a tabular classifier.

Both follow the same protocol used by every matcher in this project:

    m = Matcher(...)        # configured, no data yet
    m.fit(train_pair_table) # learns a threshold / trains on features
    m.score(pairs)          # -> probabilities in [0, 1]

Keeping ``score`` continuous (rather than a hard 0/1) is deliberate: the
calibration and threshold-sweep analyses need the full ranking, and the paper
tables report the chosen threshold separately.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from ..features import build_features, feature_group
from ..preprocess import DEFAULT_LEXICON, AttributeLexicon, normalize
from ..schema import Dataset, PairTable


class Matcher(Protocol):
    def fit(self, train: PairTable) -> "Matcher": ...
    def score(self, pairs: pd.DataFrame) -> np.ndarray: ...


class SimilarityMatcher:
    """Threshold on a single normalized string similarity (the no-ML baseline).

    Args:
        metric: one of ``jaro_winkler``, ``ratio``, ``token_sort``,
            ``token_set``, ``containment``.
        threshold: accept above this value. If None, :meth:`fit` learns it by
            maximizing F1 on the training candidates.
    """

    def __init__(self, metric: str = "token_set", threshold: float | None = None) -> None:
        self.metric = metric
        self.threshold = threshold
        self.dataset: Dataset | None = None

    def bind(self, ds: Dataset) -> SimilarityMatcher:
        self.dataset = ds
        return self

    def _sim(self, a: str, b: str) -> float:
        from rapidfuzz import distance as rf_distance

        na, nb = normalize(a), normalize(b)
        if self.metric == "jaro_winkler":
            # rapidfuzz >=3: Jaro-Winkler lives in rapidfuzz.distance and is
            # already on [0,1]; the fuzz.* functions are on [0,100].
            return rf_distance.JaroWinkler.similarity(na, nb)
        if self.metric == "levenshtein":
            return rf_distance.Levenshtein.normalized_similarity(na, nb)
        if self.metric == "ratio":
            return fuzz.ratio(na, nb) / 100.0
        if self.metric == "token_sort":
            return fuzz.token_sort_ratio(na, nb) / 100.0
        if self.metric == "token_set":
            return fuzz.token_set_ratio(na, nb) / 100.0
        if self.metric == "containment":
            ta, tb = set(na.split()), set(nb.split())
            if not ta or not tb:
                return 0.0
            small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
            return len(ta & tb) / len(small)
        raise ValueError(f"unknown metric {self.metric!r}")

    def score(self, pairs: pd.DataFrame) -> np.ndarray:
        assert self.dataset is not None, "call bind(ds) first"
        texts = self.dataset.df.set_index(self.dataset.df["record_id"].astype(str))["text"].astype(str)
        la = texts.reindex(pairs["record_id_l"].astype(str)).to_numpy()
        rb = texts.reindex(pairs["record_id_r"].astype(str)).to_numpy()
        return np.array([self._sim(str(a), str(b)) for a, b in zip(la, rb, strict=True)], dtype=float)

    def fit(self, train: PairTable) -> SimilarityMatcher:
        if self.threshold is not None:
            return self
        df = train.labeled.frame
        if df.empty:
            self.threshold = 0.85
            return self
        s = self.score(df)
        y = df["label"].astype(int).to_numpy()
        best = (float("-inf"), 0.85)
        for t in np.unique(np.concatenate(([0.0], s, [1.0 + 1e-9]))):
            pred = (s >= t).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            if f1 > best[0]:
                best = (f1, float(t))
        self.threshold = best[1]
        return self


class ClassifierMatcher:
    """Supervised matcher over the feature families (the main workhorse).

    Uses histogram gradient boosting by default: it is strong on tabular
    similarity features, handles missing attributes natively, trains in seconds
    on a laptop, and is a fair comparison point to published feature-based
    matchers (DeepMatcher's best classic configuration was also GBDT).

    Args:
        feature_groups: which feature families to include. Controls the ablation
            (Objective 4) without touching model code:
            ``{"fuzz", "tok", "attr", "num", "emb"}`` subsets.
        exclude_attr_features: drop specific ``attr_*`` columns, used to test
            which single attribute drives the variant-disambiguation gain.
    """

    def __init__(
        self,
        ds: Dataset,
        feature_groups: set[str] | None = None,
        model: str = "gbdt",
        lexicon: AttributeLexicon = DEFAULT_LEXICON,
        embeddings: np.ndarray | None = None,
        random_state: int = 0,
    ) -> None:
        self.ds = ds
        self.groups = feature_groups or {"fuzz", "tok", "attr", "num"}
        self.model_kind = model
        self.lexicon = lexicon
        self.embeddings = embeddings
        self.random_state = random_state
        self._cols: list[str] = []
        self._est = None

    # ---------------------------------------------------------------- helpers
    def _features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        feats = build_features(
            pairs["record_id_l"].astype(str),
            pairs["record_id_r"].astype(str),
            self.ds,
            lexicon=self.lexicon,
            embeddings=self.embeddings,
        )
        keep = [c for c in feats.columns if feature_group(c) in self.groups]
        return feats[keep]

    # ------------------------------------------------------------------- api
    def fit(self, train: PairTable) -> ClassifierMatcher:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = self._features(train.labeled.frame)
        y = train.labeled.frame["label"].astype(int).to_numpy()
        self._cols = list(X.columns)
        if len(np.unique(y)) < 2:
            raise ValueError("training candidates contain only one class")

        if self.model_kind == "gbdt":
            self._est = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.1, random_state=self.random_state
            )
        elif self.model_kind == "logreg":
            self._est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        else:
            raise ValueError(f"unknown model {self.model_kind!r}")
        self._est.fit(X.to_numpy(), y)
        return self

    def score(self, pairs: pd.DataFrame) -> np.ndarray:
        assert self._est is not None, "call fit() before score()"
        X = self._features(pairs)[self._cols]
        return self._est.predict_proba(X.to_numpy())[:, 1]

    def feature_importance(self) -> pd.Series:
        """Gain-free importances for the paper's "what carried the signal" table."""
        if self._est is None or not hasattr(self._est, "feature_importances_"):
            return pd.Series(dtype=float)
        return pd.Series(self._est.feature_importances_, index=self._cols).sort_values(ascending=False)
