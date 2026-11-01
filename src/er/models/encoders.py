"""Embedding encoders (bi-encoder stage).

Text embeddings come from Sentence Transformers and are cached on disk keyed by
``(model, dataset fingerprint)``. Caching matters for research: an embedding run
over 100k records takes minutes, and every threshold/ablation experiment should
reuse the exact same vectors rather than re-encode them and introduce noise.

An image encoder is implemented but **not used in the first paper**: the dataset
audit found no 2024-2026 multimodal product-matching benchmark with a licence
clean enough for publication (see docs/DATASETS.md). It is wired up so the
multimodal hypothesis can be tested the moment an image-bearing, licensed catalog
is available, without rewriting the matcher.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_CACHE = Path.home() / ".cache" / "er-research"


def _cache_path(model: str, fingerprint: str, kind: str, cache_dir: Path) -> Path:
    tag = hashlib.sha256(f"{model}|{fingerprint}|{kind}".encode()).hexdigest()[:16]
    return cache_dir / f"{kind}_{tag}.npy"


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def encode_texts(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    fingerprint: str = "",
    cache_dir: Path = DEFAULT_CACHE,
    batch_size: int = 128,
    use_cache: bool = True,
) -> np.ndarray:
    """Encode texts to L2-normalised embeddings, with disk caching.

    Args:
        fingerprint: :attr:`er.schema.Dataset.fingerprint`; required for cache
            correctness. Without it a changed dataset would silently reuse old
            vectors.
    """
    cache_dir = Path(cache_dir)
    path = _cache_path(model_name, fingerprint, "text", cache_dir)
    if use_cache and path.exists():
        return np.load(path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=_device())
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 2000,
        convert_to_numpy=True,
    ).astype("float32")

    if fingerprint:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
    return emb


def encode_images(
    image_paths_or_urls: Sequence[str],
    model_name: str = "clip-ViT-B-32",
    fingerprint: str = "",
    cache_dir: Path = DEFAULT_CACHE,
    batch_size: int = 64,
) -> np.ndarray:
    """CLIP image embeddings. Kept for the multimodal stage (blocked on licensing).

    Raises:
        NotImplementedError: the text-only datasets used by the first paper have
            no images. Replace the body with ``sentence_transformers.CLIPModel``
            once an image-bearing, licence-cleared catalog is available.
    """
    raise NotImplementedError(
        "Image encoding is intentionally disabled until a multimodal dataset with a "
        "publication-clean licence is selected. See docs/DATASETS.md (section: image data)."
    )


class EmbeddingMatcher:
    """Cosine similarity over Sentence Transformer embeddings.

    The single strongest baseline in the literature for text-only product
    matching, and the base matcher our attribute constraints are compared
    against. Score is mapped from cosine to [0, 1] so that the same threshold
    semantics apply across every matcher in the tables.

    Args:
        dim_reduce: optional PCA dim. Reduces FAISS cost; must be *fit on train
            only*, otherwise the test set leaks into the representation.
    """

    def __init__(
        self,
        ds,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim_reduce: int | None = None,
        train_mask: np.ndarray | None = None,
        cache_dir: Path = DEFAULT_CACHE,
    ) -> None:
        self.ds = ds
        self.model_name = model_name
        self.dim_reduce = dim_reduce
        self.train_mask = train_mask
        self.cache_dir = cache_dir
        self._emb: np.ndarray | None = None
        self._pca = None

    # ------------------------------------------------------------------ state
    @property
    def embeddings(self) -> np.ndarray:
        if self._emb is None:
            self._emb = encode_texts(
                self.ds.df["text"].astype(str).tolist(),
                model_name=self.model_name,
                fingerprint=self.ds.fingerprint(),
                cache_dir=self.cache_dir,
            )
            if self.dim_reduce:
                from sklearn.decomposition import PCA

                fit_data = (
                    self._emb[self.train_mask]
                    if self.train_mask is not None and self.train_mask.sum() > self.dim_reduce
                    else self._emb
                )
                self._pca = PCA(n_components=self.dim_reduce, random_state=0)
                self._pca.fit(fit_data)
                self._emb = self._pca.transform(self._emb).astype("float32")
                norms = np.linalg.norm(self._emb, axis=1, keepdims=True)
                self._emb = self._emb / np.maximum(norms, 1e-9)
        return self._emb

    def bind(self, ds) -> EmbeddingMatcher:
        self.ds = ds
        self._emb = None
        return self

    # -------------------------------------------------------------------- api
    def fit(self, train) -> EmbeddingMatcher:
        _ = self.embeddings  # warm the cache so scoring is not timed inside fit
        return self

    def score(self, pairs) -> np.ndarray:
        emb = self.embeddings
        pos = {str(r): i for i, r in enumerate(self.ds.df["record_id"].astype(str))}
        ia = np.fromiter((pos[str(x)] for x in pairs["record_id_l"]), dtype=int, count=len(pairs))
        ib = np.fromiter((pos[str(x)] for x in pairs["record_id_r"]), dtype=int, count=len(pairs))
        cos = np.einsum("ij,ij->i", emb[ia], emb[ib])
        return np.clip((cos + 1.0) / 2.0, 0.0, 1.0)

    def cosine(self, pairs) -> np.ndarray:
        """Raw cosine, for the Recall@K table where [0,1] remapping is misleading."""
        emb = self.embeddings
        pos = {str(r): i for i, r in enumerate(self.ds.df["record_id"].astype(str))}
        ia = np.fromiter((pos[str(x)] for x in pairs["record_id_l"]), dtype=int, count=len(pairs))
        ib = np.fromiter((pos[str(x)] for x in pairs["record_id_r"]), dtype=int, count=len(pairs))
        return np.einsum("ij,ij->i", emb[ia], emb[ib])
