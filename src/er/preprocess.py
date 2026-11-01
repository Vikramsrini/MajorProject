"""Text normalization and attribute extraction.

Two responsibilities:

1. :func:`normalize` - a conservative, deterministic normalization used before
   any similarity computation. Deliberately *not* aggressive: over-normalizing
   destroys the signals (variant markers such as ``Plus``/``Pro``, storage sizes)
   that the attribute-aware matcher depends on.
2. :func:`extract_attributes` - a rule-based extractor for the attributes that
   act as hard constraints in the matcher (H2). It is intentionally auditable so
   that its precision/recall can be reported on a hand-labeled sample, which is
   Objective 1 of the project.

The extractor is dictionary-light on purpose: brand/variant lists are supplied
per-dataset through :class:`AttributeLexicon` so results transfer across domains
without hidden dataset-specific tuning.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #

_PUNCT = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")
_SEPARATORS = re.compile(r"\s*[,;/|]\s*")

#: Abbreviations that recur in product titles across marketplaces.
_ABBREV: dict[str, str] = {
    "w/": "with",
    "wr": "with",
    "approx": "approximately",
    "oz": "oz",
    "pcs": "pieces",
    "pkg": "pack",
    "bx": "box",
    "un": "unit",
    "gn": "gen",
    "gbyt": "gb",
    "tb": "tb",
    "mbps": "mbps",
}

#: Marketing noise that should not influence similarity.
_STOPWORDS = {
    "new",
    "original",
    "genuine",
    "authentic",
    "bulk",
    "sale",
    "free",
    "shipping",
    "best",
    "bestseller",
    "hot",
    "price",
    "offer",
    "with",
    "for",
    "the",
    "and",
    "of",
    "in",
    "a",
    "an",
    "pcs",
    "pack",
}


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def normalize(text: str, *, keep_numerics: bool = True, drop_stopwords: bool = False) -> str:
    """Lowercase, de-accent, unify separators and punctuation.

    Args:
        text: raw title/description.
        keep_numerics: if False, digits are removed (used only by blocking that
            must ignore sizes, e.g. model-family blocking).
        drop_stopwords: remove marketing filler. Off by default for matching
            features because dropping tokens changes IDF weighting.
    """
    if text is None:
        return ""
    s = strip_accents(str(text)).lower()
    s = unicodedata.normalize("NFKC", s)
    s = _SEPARATORS.sub(" ", s)
    s = s.replace("&", " and ")
    for src, dst in _ABBREV.items():
        s = re.sub(rf"(?<!\w){re.escape(src)}(?!\w)", dst, s)
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    if not keep_numerics:
        s = re.sub(r"\d+", " ", s)
        s = _WS.sub(" ", s).strip()
    if drop_stopwords:
        toks = [t for t in s.split() if t not in _STOPWORDS]
        s = " ".join(toks)
    return s


def tokens(text: str) -> list[str]:
    return normalize(text).split()


# --------------------------------------------------------------------------- #
# attribute extraction
# --------------------------------------------------------------------------- #:

#: Attribute names in canonical reporting order.
ATTRIBUTE_NAMES = ("brand", "model_family", "variant", "capacity", "color", "quantity", "year")

_CAPACITY = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(gb|tb|mb|kb|g|t|terabyte|gigabyte|mah|mah)\b", re.I
)
_STORAGE_ALIASES = {"g": "gb", "t": "tb", "gigabyte": "gb", "terabyte": "tb"}
_COLOR = re.compile(
    r"(?<!\w)(black|white|silver|grey|gray|space\s+gray|space\s+grey|gold|rose\s+gold|"
    r"midnight|starlight|blue|red|green|pink|purple|yellow|orange|brown|beige|"
    r"graphite|titanium|indigo|navy|crimson|mint)\b",
    re.I,
)
_QUANTITY = re.compile(r"(?<![\w.])(\d{1,4})\s*(x|pack|pcs|pieces|units|count|ea)\b", re.I)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
#: A trailing size/designator glued to a model token, e.g. "iphone15", "note20".
_GLUED_MODEL = re.compile(r"^([a-z]{3,})(\d{1,3})$")


@dataclass
class AttributeLexicon:
    """Dataset-supplied vocabularies for brand and variant detection.

    Attributes:
        brands: known brand tokens, matched case-insensitively.
        variants: designator tokens that mark a *different* product when they
            disagree (Plus, Pro Max, Mini, Ultra, II, 5G, ...). These are the
            hard-negative signal for near-identical listings.
        model_prefixes: words that begin a model family (iphone, galaxy, kindle).
    """

    brands: set[str] = field(default_factory=set)
    variants: set[str] = field(default_factory=set)
    model_prefixes: set[str] = field(default_factory=set)

    @classmethod
    def from_iterables(
        cls,
        brands: Iterable[str] = (),
        variants: Iterable[str] = (),
        model_prefixes: Iterable[str] = (),
    ) -> AttributeLexicon:
        return cls(
            brands={b.lower() for b in brands},
            variants={v.lower() for v in variants},
            model_prefixes={m.lower() for m in model_prefixes},
        )


#: A deliberately small, domain-neutral default. Override per dataset.
DEFAULT_LEXICON = AttributeLexicon.from_iterables(
    brands=["apple", "samsung", "google", "oneplus", "sony", "microsoft", "dell", "lenovo", "hp"],
    variants=[
        "plus",
        "pro",
        "promax",
        "pro max",
        "ultra",
        "mini",
        "lite",
        "max",
        "se",
        "air",
        "note",
        "fold",
        "flip",
        "ii",
        "iii",
        "iv",
        "v",
        "5g",
        "4g",
        "wifi",
        "cellular",
    ],
    model_prefixes=["iphone", "ipad", "galaxy", "pixel", "kindle", "macbook", "note"],
)


def _canonical_capacity(value: str, unit: str) -> str:
    unit = _STORAGE_ALIASES.get(unit.lower(), unit.lower())
    num = float(value.replace(",", ""))
    # Normalise storage to GB so 1TB and 1024GB are comparable, mAh kept as-is.
    if unit == "tb":
        num *= 1024
        unit = "gb"
    return f"{num:g}{unit}"


def extract_attributes(text: str, lexicon: AttributeLexicon = DEFAULT_LEXICON) -> dict[str, object]:
    """Extract auditable hard-constraint attributes from a single title.

    Returns a dict with keys from :data:`ATTRIBUTE_NAMES`; absent attributes are
    omitted rather than set to None, so the matcher can distinguish "unknown"
    from "present and conflicting". That distinction is required for the missing
    attribute ablation (Objective 4).
    """
    norm = normalize(text)
    toks = norm.split()
    tokset = set(toks)
    out: dict[str, object] = {}

    brand_hits = sorted(tokset & lexicon.brands)
    if brand_hits:
        out["brand"] = brand_hits[0]

    variant_hits = sorted({v for v in lexicon.variants if v in tokset or f" {v} " in f" {norm} "})
    if variant_hits:
        # Multi-word variants ("pro max") collapse to their canonical form.
        out["variant"] = " ".join(variant_hits)

    cap = _CAPACITY.search(norm)
    if cap:
        out["capacity"] = _canonical_capacity(cap.group(1), cap.group(2))
    else:
        glued = next((m for m in map(_GLUED_MODEL.match, toks) if m), None)
        if glued and glued.group(1) in lexicon.model_prefixes:
            out["model_family"] = f"{glued.group(1)} {glued.group(2)}"

    col = _COLOR.search(norm)
    if col:
        out["color"] = _WS.sub(" ", col.group(1).lower())

    qty = _QUANTITY.search(norm)
    if qty:
        n = int(qty.group(1))
        out["quantity"] = str(n)

    yr = _YEAR.search(norm)
    if yr:
        out["year"] = yr.group(1)

    # model family: prefix token followed by a number token, e.g. "iphone 15".
    for i, t in enumerate(toks[:-1]):
        if t in lexicon.model_prefixes and re.fullmatch(r"\d{1,3}", toks[i + 1] or ""):
            out["model_family"] = f"{t} {toks[i + 1]}"
            break

    return out


#: Units that are meaningful when separated from the number by a space ("4 TB").
#: Adjacent letters always count as a unit ("4TB", "50mm", "st4000ne001"), which
#: is what also lets model codes such as "ne001" vs "ne002" be compared.
_KNOWN_SPACED_UNITS = {
    "mm", "cm", "km", "m", "in", "inch", "ft", "gb", "tb", "mb", "kb", "mah", "wh",
    "w", "kw", "v", "mv", "hz", "khz", "mhz", "ghz", "rpm", "nit", "nits", "mp",
    "px", "ml", "l", "dl", "oz", "kg", "g", "lb", "lbs", "bps", "mbps", "gbps",
    "tflops", "gflops", "ah", "va", "db", "dbi", "atm", "bar", "psi", "gal",
}

#: Keys that are numeric noise rather than product identity. Note that ``mm`` is
#: deliberately *not* here: for camera lenses a focal length is identity ("50mm"
#: vs "85mm" are different products), which is exactly the WDC Products mix.
_SPEC_STOPKEYS = {
    "days", "day", "years", "year", "months", "weeks", "pcs", "pack", "pieces",
    "with", "for", "the", "and", "x", "or", "of", "to", "at", "by", "in", "on",
}

_SPEC_RE = re.compile(
    r"(?:(?P<pre>[a-z]{1,6})[ \t]*/?[ \t]*)?(?P<num>\d+(?:[.,]\d+)?)"
    r"(?P<gap>[ \t]?)(?P<post>[a-z]{1,6})?"
)


def specifier_map(text: str) -> dict[str, set[str]]:
    """Extract unit-qualified numeric specifiers as ``{unit: {values}}``.

    Regex attribute extraction cannot enumerate every product code, but a
    *disagreement* is detectable without knowing them: two offers of the same
    product rarely state different values for the same unit ("50mm" vs "85mm",
    "f/1.8" vs "f/1.4", "4TB" vs "8TB"), while cross-source title noise almost
    always preserves the number-unit pairing. Keying on the unit rather than the
    whole token makes the comparison robust to spacing, slashes and word order.

    A trailing word only counts as a unit when it is directly attached to the
    number or is in :data:`_KNOWN_SPACED_UNITS`; without that rule "f/1.8 lens"
    invents a unit called "lens" and manufactures conflicts.
    """
    s = strip_accents(str(text or "")).lower()
    s = re.sub(r"[,;()\[\]{}\"']", " ", s)
    out: dict[str, set[str]] = {}
    for m in _SPEC_RE.finditer(s):
        pre = (m.group("pre") or "").strip()
        post = (m.group("post") or "").strip()
        gap = m.group("gap") or ""
        raw = m.group("num").replace(",", ".")
        if post and gap and post not in _KNOWN_SPACED_UNITS:
            post = ""  # a separated non-unit word: "1.8 lens", "2 with bag"
        key = post or pre
        if not key or not (pre or post) or key in _SPEC_STOPKEYS:
            continue
        try:
            val = f"{float(raw):g}"
        except ValueError:
            continue
        out.setdefault(key, set()).add(val)
    return out


def specifier_conflicts(a: dict[str, set[str]], b: dict[str, set[str]]) -> list[str]:
    """Units stated on both sides whose value sets do not intersect.

    One-sided keys are *not* conflicts: an omitted specification is missing
    evidence, and treating absence as contradiction is what turns a constraint
    into a recall killer on real catalogs.
    """
    out = []
    for key in set(a) & set(b):
        if a[key] and b[key] and not (a[key] & b[key]):
            out.append(f"specifier:{key}")
    return out


def combined_attributes(
    text: str, stated: dict | None, lexicon: AttributeLexicon = DEFAULT_LEXICON
) -> dict[str, object]:
    """Merge inferred and stated attributes: text extraction first, source wins.

    Both sources must always be consulted. Taking *only* the stated dict (which
    for WDC Products is brand and price) would silently disable variant and
    capacity detection, i.e. the whole contribution, on any dataset that ships
    partial structured fields. Stated values override inferred ones because a
    catalog column is more reliable than a title regex.
    """
    attrs = extract_attributes(text, lexicon)
    if stated:
        for key, val in stated.items():
            if val in (None, "") or str(val).lower() in {"nan", "none"}:
                continue
            attrs[key] = normalize(str(val)) or str(val)
    return attrs


def attributes_from_fields(
    fields: dict[str, str], lexicon: AttributeLexicon = DEFAULT_LEXICON
) -> dict[str, object]:
    """Merge structured columns and free-text extraction.

    Structured values win when present (they are what a catalog would trust);
    free text fills the gaps. Both sources are recorded so the matcher can tell
    whether an attribute was stated or inferred.
    """
    text = " ".join(str(v) for v in fields.values() if v not in (None, ""))
    attrs = extract_attributes(text, lexicon)
    for key in ATTRIBUTE_NAMES:
        val = fields.get(key)
        if val not in (None, "") and str(val).lower() != "nan":
            attrs[key] = normalize(str(val)) or str(val)
    return attrs


def agreement(a: dict[str, object], b: dict[str, object]) -> dict[str, float]:
    """Per-attribute agreement signals between two records.

    Emits three values per attribute in [-1, 1]:
    ``+1`` both present and equal, ``-1`` both present and different,
    ``0`` at least one side missing (i.e. no evidence, not agreement).

    Encoding "missing" as 0 rather than -1 is what lets the classifier learn to
    tolerate absent specifications instead of rejecting them.
    """
    out: dict[str, float] = {}
    for name in ATTRIBUTE_NAMES:
        va, vb = a.get(name), b.get(name)
        if va in (None, "") or vb in (None, ""):
            out[f"attr_{name}"] = 0.0
            out[f"attr_{name}_known"] = 0.0 if va in (None, "") and vb in (None, "") else 1.0
        else:
            out[f"attr_{name}"] = 1.0 if str(va) == str(vb) else -1.0
            out[f"attr_{name}_known"] = 1.0
    return out
