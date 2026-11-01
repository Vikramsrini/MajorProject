"""Cross-encoder matcher (Ditto-style), for GPU fine-tuning on Colab/Kaggle.

Why this exists: every scoring-stage claim in this project is currently made on a base
matcher ~15 F1 points below the *published simple baseline*, which is the first thing a
reviewer will attack. The benchmark reports fine-tuned transformers at 0.67-0.73 F1 on
these splits, so this module reproduces that family closely enough to be comparable and
then re-tests the constraint against a strong base.

Design decisions that matter for the paper:

* **Field-level serialisation.** Records are serialised as
  ``[COL] title [VAL] ... [COL] description [VAL] ... [COL] brand [VAL] ...`` and the two
  records are passed as a tokenizer *sentence pair*. The published RoBERTa and Ditto
  numbers use this scheme; feeding flattened text would make the comparison meaningless.
* **Early stopping on a slice of TRAIN, never on validation.** The validation split is
  already used for temperature scaling and for choosing the operating point. Reusing it
  for early stopping would tune on the same data that sets the decision threshold.
* **A step budget (``max_steps``).** GPU sessions die; a path that cannot be smoke-tested
  in 20 steps is one nobody can debug inside a notebook.
* **Checkpointing (config + weights) to a directory**, so a disconnected session resumes
  instead of retraining.

A plain PyTorch loop, with no ``transformers.Trainer`` and no ``accelerate``/``datasets``
imports: that keeps it working across the differing library versions on Colab and Kaggle
and makes the training semantics explicit instead of framework-dependent.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import PairTable

#: Fields serialised into the model input, in order. Missing fields are skipped.
SERIAL_FIELDS: tuple[str, ...] = ("title", "description", "brand")


def _cell_str(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def serialize_record(row: dict | pd.Series) -> str:
    """Turn one record into Ditto-style ``[COL] field [VAL] value`` text.

    Falls back to the flattened ``text`` column when no structured field is present,
    so the matcher still works on datasets loaded without title/description columns.
    """
    parts: list[str] = []
    for field in SERIAL_FIELDS:
        if field == "brand":
            attrs = row.get("attrs") if hasattr(row, "get") else None
            val = attrs.get("brand") if isinstance(attrs, dict) else None
        else:
            val = row.get(field)
        s = _cell_str(val)
        if s:
            parts.append(f"[COL] {field} [VAL] {s}")
    if not parts:
        fallback = _cell_str(row.get("text")) if hasattr(row, "get") else None
        if fallback:
            parts.append(f"[COL] text [VAL] {fallback}")
    return " ".join(parts)


def _max_f1_at_best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Early-stopping criterion: best achievable F1 over thresholds.

    Using F1 at a fixed 0.5 would stop training at a model whose *ranking* is fine but
    whose calibration is still shifting, which is exactly the phase early stopping should
    survive.

    Cuts are evaluated only at score boundaries, with all tied items accepted together,
    because a threshold can only compare scores. Evaluating every prefix instead (the
    obvious implementation) silently lets the cut pick *which* of several equally-scored
    items to accept: with four candidates all at 0.5 and one positive it reports F1 1.0
    where the best realisable cut is 0.4, which inflates the stopping signal.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    if len(y) == 0 or n_pos == 0:
        return 0.0

    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    # last index of each tie block = the only places a threshold can actually land
    ends = np.r_[np.where(ss[1:] != ss[:-1])[0], len(ss) - 1]
    tp_e, fp_e = tp[ends].astype(float), fp[ends].astype(float)
    fn_e = n_pos - tp_e
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = tp_e / np.maximum(1.0, tp_e + fp_e)
        rec = tp_e / n_pos
        f1 = 2 * prec * rec / np.maximum(1e-12, prec + rec)
    return float(np.nanmax(f1))


def _load_tokenizer(model_name: str):
    """Load a tokenizer, falling back to the slow implementation.

    Not every checkpoint ships a fast tokenizer, and converting a slow one needs
    ``sentencepiece``/``tiktoken``. On a rented GPU session that dependency is often
    absent, and the failure surfaces only after the data has been downloaded. Trying
    ``use_fast=True`` first and falling back keeps the module usable for models such as
    ``microsoft/deberta-v3-*`` (optional dependency) without a hard requirement.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, then retried slow
        print(f"[crossencoder] fast tokenizer unavailable ({type(exc).__name__}); using slow tokenizer")
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


class CrossEncoderMatcher:
    """Fine-tuned cross-encoder over serialised record pairs.

    Args:
        ds: dataset holding the records (needs ``title``/``description`` ideally).
        model_name: HuggingFace encoder with a classification head, e.g.
            ``roberta-base`` (as in the benchmark paper) or ``microsoft/deberta-v3-small``.
        max_steps: cap on optimiser steps; set a small value to smoke-test on CPU.
        es_fraction: share of TRAIN held out for early stopping (validation is untouched).
        fp16: mixed precision; default on when CUDA is available, off on CPU/MPS.
    """

    def __init__(
        self,
        ds,
        model_name: str = "roberta-base",
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 2e-5,
        max_len: int = 256,
        seed: int = 0,
        device: str | None = None,
        max_steps: int | None = None,
        es_fraction: float = 0.12,
        warmup_frac: float = 0.1,
        fp16: bool | None = None,
        eval_every: int = 200,
        verbose: bool = True,
    ) -> None:
        self.ds = ds
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_len = max_len
        self.seed = seed
        self.max_steps = max_steps
        self.es_fraction = es_fraction
        self.warmup_frac = warmup_frac
        self.eval_every = eval_every
        self.verbose = verbose
        self._device_arg = device
        self._fp16_arg = fp16
        self._model = None
        self._tok = None
        self._docs: dict[str, str] | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ setup
    @property
    def device(self) -> str:
        if self._device_arg:
            return self._device_arg
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def use_fp16(self) -> bool:
        if self._fp16_arg is not None:
            return self._fp16_arg
        return self.device == "cuda"

    def bind(self, ds) -> CrossEncoderMatcher:
        self.ds = ds
        self._docs = None
        return self

    def _doc_map(self) -> dict[str, str]:
        if self._docs is None:
            df = self.ds.df
            self._docs = {
                str(r["record_id"]): serialize_record(r)
                for _, r in df.iterrows()
            }
        return self._docs

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification

        self._tok = _load_tokenizer(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        ).to(self.device)

    def _pairs_text(self, pairs: pd.DataFrame) -> tuple[list[str], list[str]]:
        docs = self._doc_map()
        left = [docs.get(str(x), "") for x in pairs["record_id_l"]]
        right = [docs.get(str(x), "") for x in pairs["record_id_r"]]
        return left, right

    def _batched_probs(self, pairs: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
        """Inference only: no dropout, no grad, deterministic batch order."""
        import torch

        self._ensure_model()
        assert self._model is not None and self._tok is not None
        self._model.eval()
        left, right = self._pairs_text(pairs)
        out = np.zeros(len(left), dtype=float)
        with torch.no_grad():
            for start in range(0, len(left), batch_size):
                sl = slice(start, start + batch_size)
                enc = self._tok(
                    left[sl], right[sl], truncation=True, max_length=self.max_len,
                    padding=True, return_tensors="pt",
                ).to(self.device)
                logits = self._model(**enc).logits
                out[sl] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        return out

    # -------------------------------------------------------------------- api
    def score(self, pairs: pd.DataFrame) -> np.ndarray:
        return self._batched_probs(pairs)

    def fit(self, train: PairTable) -> CrossEncoderMatcher:
        import torch
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        df = train.labeled.frame.reset_index(drop=True)
        if df.empty:
            raise ValueError("no labeled training pairs")
        # deterministic split of TRAIN for early stopping; validation stays clean
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(df))
        n_es = max(1, int(round(self.es_fraction * len(df))))
        es_idx, fit_idx = idx[:n_es], idx[n_es:]
        if self.verbose:
            print(
                f"[crossencoder] {self.model_name} on {self.device} "
                f"(fp16={self.use_fp16}); fit={len(fit_idx)} es={n_es} "
                f"epochs={self.epochs} bs={self.batch_size} max_steps={self.max_steps}"
            )

        self._ensure_model()
        assert self._model is not None and self._tok is not None and self._model is not None

        docs = self._doc_map()
        l_col, r_col, y_col = "record_id_l", "record_id_r", "label"
        left_all = [docs.get(str(x), "") for x in df[l_col]]
        right_all = [docs.get(str(x), "") for x in df[r_col]]
        y_all = df[y_col].astype(int).to_numpy()

        steps_per_epoch = max(1, math.ceil(len(fit_idx) / self.batch_size))
        total_steps = steps_per_epoch * self.epochs
        if self.max_steps:
            total_steps = min(total_steps, self.max_steps)
        warmup = max(1, int(self.warmup_frac * total_steps))
        opt = AdamW(self._model.parameters(), lr=self.lr)
        sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16) if hasattr(torch.amp, "GradScaler") else None
        lossf = torch.nn.CrossEntropyLoss()

        best_f1, best_state, best_step = -1.0, None, -1
        step, t0 = 0, time.time()
        done = False
        for epoch in range(self.epochs):
            self._model.train()
            order = rng.permutation(fit_idx)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                enc = self._tok(
                    [left_all[i] for i in batch], [right_all[i] for i in batch],
                    truncation=True, max_length=self.max_len, padding=True, return_tensors="pt",
                ).to(self.device)
                labels = torch.tensor(y_all[batch], dtype=torch.long, device=self.device)
                opt.zero_grad(set_to_none=True)
                if self.use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self._model(**enc).logits
                        loss = lossf(logits, labels)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:  # pragma: no cover - defensive for odd torch versions
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                        opt.step()
                else:
                    logits = self._model(**enc).logits
                    loss = lossf(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    opt.step()
                sched.step()
                step += 1

                if step % self.eval_every == 0 or step == total_steps:
                    es_pairs = pd.DataFrame(
                        {"record_id_l": df[l_col].iloc[es_idx].to_numpy(),
                         "record_id_r": df[r_col].iloc[es_idx].to_numpy()}
                    )
                    probs = self._batched_probs(es_pairs)
                    f1 = _max_f1_at_best_threshold(y_all[es_idx], probs)
                    self.history.append(
                        {"step": step, "epoch": epoch + 1, "loss": float(loss.detach().cpu()),
                         "es_f1": f1, "seconds": round(time.time() - t0, 1)}
                    )
                    if self.verbose:
                        print(f"  step {step:5d}/{total_steps}  loss {float(loss.detach()):.4f}  es_f1 {f1:.4f}"
                              f"  ({time.time()-t0:.0f}s)")
                    if f1 > best_f1:
                        best_f1, best_step = f1, step
                        best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                    self._model.train()
                if self.max_steps and step >= self.max_steps:
                    done = True
                    break
            if done:
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
            if self.verbose:
                print(f"[crossencoder] restored best checkpoint at step {best_step} (es_f1={best_f1:.4f})")
        self.best_es_f1 = best_f1
        self.trained_steps = step
        return self

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path | str) -> Path:
        """Save weights + config so a disconnected notebook session can resume."""
        import torch

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self._ensure_model()
        assert self._model is not None
        torch.save(self._model.state_dict(), d / "model.pt")
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name, "max_len": self.max_len,
                    "serial_fields": list(SERIAL_FIELDS), "seed": self.seed,
                    "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr,
                    "best_es_f1": getattr(self, "best_es_f1", None),
                },
                indent=2,
            )
        )
        if self.history:
            pd.DataFrame(self.history).to_csv(d / "history.csv", index=False)
        return d

    def load(self, directory: Path | str) -> CrossEncoderMatcher:
        import torch
        from transformers import AutoModelForSequenceClassification

        d = Path(directory)
        cfg = json.loads((d / "config.json").read_text())
        self.model_name = cfg.get("model_name", self.model_name)
        self.max_len = int(cfg.get("max_len", self.max_len))
        self._tok = _load_tokenizer(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        ).to(self.device)
        self._model.load_state_dict(torch.load(d / "model.pt", map_location=self.device))
        self._model.eval()
        return self
