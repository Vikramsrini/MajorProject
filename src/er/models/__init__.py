"""Matchers.

Every matcher implements the same two-method protocol so the experiment runner
can treat them interchangeably in tables and ablations::

    fit(train_pairs)      -> self
    score(pairs)          -> ndarray in [0, 1]

* :class:`~er.models.baselines.SimilarityMatcher` - thresholded string score
  (the baseline every ER paper includes).
* :class:`~er.models.baselines.ClassifierMatcher` - gradient-boosted / logistic
  model over the feature families.
* :class:`~er.models.encoders.EmbeddingMatcher` - Sentence Transformer cosine.
* :class:`~er.models.attribute.AttributeConstrainedMatcher` - the proposed
  contribution: wraps any matcher and enforces attribute conflicts.
"""

from .baselines import ClassifierMatcher, SimilarityMatcher
from .attribute import AttributeConstrainedMatcher, hard_conflicts
from .encoders import EmbeddingMatcher, encode_texts

__all__ = [
    "AttributeConstrainedMatcher",
    "ClassifierMatcher",
    "EmbeddingMatcher",
    "SimilarityMatcher",
    "encode_texts",
    "hard_conflicts",
]
