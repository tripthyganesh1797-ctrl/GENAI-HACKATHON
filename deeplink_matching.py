"""
deeplink_matching.py — Shared deeplink retrieval used by BOTH pipeline.py
(the live LLM-path Stage 2) and eval/matchers.py (the ablation study), so
the numbers in the ablation table describe the actual code that ships, not
a re-implementation.

Matches exclusively against descriptive metadata (description, message,
qna_description) — never the masked URI itself ("Matching on Masked URI
Strings" pitfall).

Two variants, deliberately kept side by side:

  * HybridDeeplinkIndex — "Variant B: Hybrid" (upgraded from the original
    keyword-Jaccard-only matcher). Blends a BM25 sparse score with a dense
    score. The dense score uses sentence-transformers embeddings when that
    package + model are available (best quality); otherwise it degrades to
    a TF-IDF -> Truncated-SVD low-rank projection, entirely offline and
    deterministic. This automatic degrade *is* the "tech-stack fallback"
    for the embedding layer specifically.

  * RulesDeeplinkIndex — "Variant A: Pure Rules-Based". Deterministic fuzzy
    keyword matching, no fitted model at all. Always available, near-zero
    cold-start cost. This is what the whole service falls back to if index
    construction fails for any reason.

Variant C ("Full-LLM Mapping") stays in eval/matchers.py since it calls out
to the LLM client, not this module.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

DUMMY_POSITIVE_DEEPLINK = "bixby://dummy_positive"

# --- Optional real embeddings ------------------------------------------
_ST_MODEL = None
_ST_TRIED = False


def _try_load_sentence_transformer():
    """Attempt to load a small sentence-transformers model once per
    process. Any failure (package missing, no network to fetch weights,
    etc.) is swallowed and we permanently fall back to TF-IDF/SVD for the
    rest of the process -- this is intentionally silent-and-safe rather
    than a hard dependency."""
    global _ST_MODEL, _ST_TRIED
    if _ST_TRIED:
        return _ST_MODEL
    _ST_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _ST_MODEL = None
    return _ST_MODEL


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class DeeplinkEntry:
    id: str
    deeplink: str
    description: str
    message: str
    qna_description: str
    original_type: str | None
    validation: dict | None

    @property
    def corpus_text(self) -> str:
        return f"{self.description} {self.message} {self.qna_description}"


def load_deeplinks(path: str | Path = "deeplinks.json") -> list[DeeplinkEntry]:
    raw = json.loads(Path(path).read_text())
    # Official file is {"_readme", "count", "deeplinks": [...]}; the
    # original placeholder file was a bare list. Support both.
    items = raw["deeplinks"] if isinstance(raw, dict) and "deeplinks" in raw else raw
    entries = []
    for i, d in enumerate(items):
        entries.append(
            DeeplinkEntry(
                id=d.get("id", f"DL-{i:04d}"),
                deeplink=d["deeplink"],
                description=d.get("description", ""),
                message=d.get("message") or "",
                qna_description=d.get("qna_description", ""),
                original_type=d.get("originalType"),
                validation=d.get("validation"),
            )
        )
    return entries


class HybridDeeplinkIndex:
    """Variant B (primary/default): BM25 + dense hybrid retrieval."""

    def __init__(self, entries: list[DeeplinkEntry], alpha: float = 0.5,
                 svd_components: int = 128):
        self.entries = [e for e in entries if e.deeplink != DUMMY_POSITIVE_DEEPLINK]
        self.alpha = alpha
        self.embedder = _try_load_sentence_transformer()

        corpus = [e.corpus_text for e in self.entries]
        self.bm25 = BM25Okapi([_tokenize(c) for c in corpus])

        if self.embedder is not None:
            self.dense_matrix = self.embedder.encode(corpus, normalize_embeddings=True)
            self.dense_kind = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            n_components = min(svd_components, max(2, len(corpus) - 1))
            self.tfidf = TfidfVectorizer(min_df=1, stop_words="english")
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.dense_matrix = self.svd.fit_transform(tfidf_matrix)
            self.dense_kind = "tfidf-svd-128 (offline fallback; sentence-transformers unavailable)"

    def _dense_query_vec(self, text: str):
        import numpy as np

        if self.embedder is not None:
            return self.embedder.encode([text], normalize_embeddings=True)
        return self.svd.transform(self.tfidf.transform([text]))

    def search(self, text: str, top_k: int = 3) -> list[tuple[DeeplinkEntry, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        if not self.entries:
            return []
        toks = _tokenize(text)
        bm25_scores = self.bm25.get_scores(toks) if toks else [0.0] * len(self.entries)
        max_bm25 = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1.0
        norm_bm25 = [s / max_bm25 for s in bm25_scores]

        dense_vec = self._dense_query_vec(text)
        dense_scores = cosine_similarity(dense_vec, self.dense_matrix)[0]
        norm_dense = [max(0.0, s) for s in dense_scores]

        combined = [self.alpha * b + (1 - self.alpha) * d
                    for b, d in zip(norm_bm25, norm_dense)]
        ranked = sorted(zip(self.entries, combined), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def best_match(self, text: str, threshold: float = 0.12) -> tuple[DeeplinkEntry | None, float]:
        results = self.search(text, top_k=1)
        if not results:
            return None, 0.0
        entry, score = results[0]
        return (entry, score) if score >= threshold else (None, score)


class RulesDeeplinkIndex:
    """Variant A (fallback): deterministic fuzzy keyword matching."""

    def __init__(self, entries: list[DeeplinkEntry]):
        self.entries = [e for e in entries if e.deeplink != DUMMY_POSITIVE_DEEPLINK]

    def best_match(self, text: str, threshold: float = 45.0) -> tuple[DeeplinkEntry | None, float]:
        best_entry, best_score = None, 0.0
        for e in self.entries:
            score = fuzz.token_set_ratio(text, e.corpus_text)
            if score > best_score:
                best_score, best_entry = score, e
        if best_entry is None or best_score < threshold:
            return None, best_score / 100.0
        return best_entry, best_score / 100.0

    def search(self, text: str, top_k: int = 3):
        scored = [(e, fuzz.token_set_ratio(text, e.corpus_text) / 100.0) for e in self.entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


_INDEX_CACHE: dict[str, HybridDeeplinkIndex | RulesDeeplinkIndex] = {}


def get_index(variant: str = "hybrid", path: str = "deeplinks.json"):
    """Process-wide singleton so pipeline.py doesn't rebuild the BM25/SVD
    index on every request."""
    key = f"{variant}:{path}"
    if key not in _INDEX_CACHE:
        entries = load_deeplinks(path)
        try:
            if variant == "rules":
                raise RuntimeError("forced rules variant")
            _INDEX_CACHE[key] = HybridDeeplinkIndex(entries)
        except Exception:
            _INDEX_CACHE[key] = RulesDeeplinkIndex(entries)
    return _INDEX_CACHE[key]


def to_title_case(text: str) -> str:
    small = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "or"}
    words = text.strip().split()
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        out.append(lw if (i > 0 and lw in small) else lw[:1].upper() + lw[1:])
    return " ".join(out)


def match_and_build_deeplink(action_name: str, steps: list[str],
                              variant: str = "hybrid",
                              path: str = "deeplinks.json") -> tuple[dict, dict | None]:
    """Used by pipeline.py's Stage 2. Returns (actionableDeeplink dict,
    validationDeeplink dict | None)."""
    index = get_index(variant, path)
    query_text = action_name + " " + " ".join(steps)
    entry, score = index.best_match(query_text)

    if entry is not None:
        actionable = {
            "deeplink": entry.deeplink,
            "description": entry.description,
            "message": entry.message,
            "originalType": entry.original_type,
        }
        validation = None
        if entry.validation:
            v = entry.validation
            validation = {
                "deeplink": v.get("deeplink", DUMMY_POSITIVE_DEEPLINK),
                "key": v.get("key", entry.message or "Setting"),
                "resultType": v.get("resultType"),
                "condition": v.get("condition"),
                "value": v.get("value"),
            }
        return actionable, validation

    return {
        "deeplink": DUMMY_POSITIVE_DEEPLINK,
        "description": f"Opens the {action_name.lower()} settings screen on the device.",
        "message": f"Open {to_title_case(action_name)}",
        "originalType": "placeholder",
    }, None
