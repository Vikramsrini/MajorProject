"""er - attribute-aware, confidence-aware entity resolution research toolkit.

Public surface used by scripts and tests. Import is kept light: torch,
sentence-transformers and faiss are pulled in lazily by the modules that need
them, so blocking/features/metrics work on a bare install.
"""

from .schema import Blocks, Dataset, PairTable
from .preprocess import (
    ATTRIBUTE_NAMES,
    AttributeLexicon,
    DEFAULT_LEXICON,
    agreement,
    extract_attributes,
    normalize,
)
from .blocking import (
    all_pairs_block,
    faiss_block,
    minhash_block,
    prefix_block,
    sorted_neighborhood_block,
    token_block,
)
from .features import build_features
from .metrics import (
    bcubed,
    best_threshold,
    blocking_quality,
    entity_pairwise_f1,
    gold_pairs_from_entities,
    pairwise_metrics,
    recall_at_k,
)
from .calibration import (
    brier_score,
    expected_calibration_error,
    reliability_table,
    selective_report,
    temperature_scale,
)
from . import io, experiment

__version__ = "0.1.0"

__all__ = [
    "ATTRIBUTE_NAMES",
    "AttributeLexicon",
    "Blocks",
    "DEFAULT_LEXICON",
    "Dataset",
    "PairTable",
    "agreement",
    "all_pairs_block",
    "best_threshold",
    "bcubed",
    "blocking_quality",
    "brier_score",
    "build_features",
    "entity_pairwise_f1",
    "experiment",
    "expected_calibration_error",
    "extract_attributes",
    "faiss_block",
    "gold_pairs_from_entities",
    "io",
    "minhash_block",
    "normalize",
    "pairwise_metrics",
    "prefix_block",
    "recall_at_k",
    "reliability_table",
    "selective_report",
    "sorted_neighborhood_block",
    "temperature_scale",
    "token_block",
]
