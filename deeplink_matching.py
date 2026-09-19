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

import feedback

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


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is",
    "it", "your", "you", "this", "that", "with", "screen", "device",
}


def _matched_keywords(text: str, entry: "DeeplinkEntry", limit: int = 5) -> list[str]:
    """Token overlap between the query and the entry's own matchable text --
    surfaced to the caller so a judge (or a developer debugging a bad match)
    can see *why* a deeplink was picked, not just that it was. Deliberately
    crude (set intersection, common stopwords dropped) to match the
    crudeness of the fuzzy/BM25 scorers themselves -- this is meant to
    explain the actual matching logic, not to be a better matcher itself."""
    q_toks = set(_tokenize(text)) - _STOPWORDS
    e_toks = set(_tokenize(entry.corpus_text)) - _STOPWORDS
    return sorted(q_toks & e_toks)[:limit]


def _pick_index_avoiding(
    ranked_indices: list[int], scores: list[float], deeplinks: list[str],
    threshold: float, avoid_deeplinks: frozenset[str] | set[str],
) -> tuple[int, bool]:
    """Task 36 (session_memory.py): shared by both index variants'
    best_match_explained(). `ranked_indices` must already be sorted
    best-first by `scores`, on whatever scale `threshold` uses. Returns
    (chosen_index, skipped_an_avoided_entry).

    When the top-ranked candidate isn't in avoid_deeplinks (the common
    case -- including every call where avoid_deeplinks is empty), this is
    just `ranked_indices[0]`, unchanged: zero behavior difference from
    before this parameter existed. Only when the top candidate IS avoided
    does this walk down to the next one that still clears `threshold` and
    isn't itself avoided. If nothing qualifies, it falls back to the
    original top candidate rather than inventing a spurious "no match" --
    see session_memory.py's module docstring for why."""
    if not ranked_indices:
        return -1, False
    top = ranked_indices[0]
    if not avoid_deeplinks or deeplinks[top] not in avoid_deeplinks:
        return top, False
    for i in ranked_indices[1:]:
        if scores[i] < threshold:
            break
        if deeplinks[i] not in avoid_deeplinks:
            return i, True
    return top, False


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

    def _component_scores(self, text: str) -> tuple[list[float], list[float]]:
        """Returns (norm_bm25, norm_dense), aligned with self.entries. Split
        out from search() so best_match_explained() can report the two raw
        components separately instead of only their blend."""
        from sklearn.metrics.pairwise import cosine_similarity

        toks = _tokenize(text)
        bm25_scores = self.bm25.get_scores(toks) if toks else [0.0] * len(self.entries)
        max_bm25 = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1.0
        norm_bm25 = [s / max_bm25 for s in bm25_scores]

        dense_vec = self._dense_query_vec(text)
        dense_scores = cosine_similarity(dense_vec, self.dense_matrix)[0]
        norm_dense = [max(0.0, s) for s in dense_scores]
        return norm_bm25, norm_dense

    def search(self, text: str, top_k: int = 3) -> list[tuple[DeeplinkEntry, float]]:
        if not self.entries:
            return []
        norm_bm25, norm_dense = self._component_scores(text)
        combined = [self.alpha * b + (1 - self.alpha) * d
                    for b, d in zip(norm_bm25, norm_dense)]
        # Adaptive re-ranking: nudge scores using accumulated human feedback
        # (see feedback.py) -- a deeplink that keeps getting thumbs-down for
        # this kind of query sinks; one that keeps getting thumbs-up rises.
        adjusted = [max(0.0, c + feedback.get_adjustment(e.deeplink))
                    for e, c in zip(self.entries, combined)]
        ranked = sorted(zip(self.entries, adjusted), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def best_match(self, text: str, threshold: float = 0.12) -> tuple[DeeplinkEntry | None, float]:
        results = self.search(text, top_k=1)
        if not results:
            return None, 0.0
        entry, score = results[0]
        return (entry, score) if score >= threshold else (None, score)

    def best_match_explained(
        self, text: str, threshold: float = 0.12, avoid_deeplinks=None
    ) -> tuple[DeeplinkEntry | None, float, dict]:
        """Same ranking as best_match(), but also returns a JSON-safe
        breakdown of *why* the top entry scored the way it did -- the raw
        BM25 and dense components before blending, the feedback nudge
        applied on top, and the overlapping keywords a person can sanity-check
        by eye. Used by match_and_build_deeplink() to surface this in the API
        response; best_match() itself is left untouched so existing callers
        (offline_fallback.py, eval/matchers.py) are unaffected.

        `avoid_deeplinks` (Task 36, session_memory.py) is an optional set
        of deeplinks to skip past in favor of the next viable candidate --
        see _pick_index_avoiding()'s docstring. None/empty (the default)
        makes this byte-identical to before that parameter existed."""
        if not self.entries:
            return None, 0.0, {"matcher": "hybrid_bm25_dense", "reason": "empty_index"}
        avoid_deeplinks = avoid_deeplinks or frozenset()

        norm_bm25, norm_dense = self._component_scores(text)
        combined = [self.alpha * b + (1 - self.alpha) * d
                    for b, d in zip(norm_bm25, norm_dense)]
        fb_adj = [feedback.get_adjustment(e.deeplink) for e in self.entries]
        adjusted = [max(0.0, c + a) for c, a in zip(combined, fb_adj)]

        ranked_indices = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)
        deeplinks = [e.deeplink for e in self.entries]
        chosen_i, skipped_avoided = _pick_index_avoiding(
            ranked_indices, adjusted, deeplinks, threshold, avoid_deeplinks
        )
        entry, score = self.entries[chosen_i], adjusted[chosen_i]

        explanation = {
            "matcher": "hybrid_bm25_dense",
            "dense_kind": self.dense_kind,
            "alpha": self.alpha,
            "bm25_component": round(norm_bm25[chosen_i], 4),
            "dense_component": round(norm_dense[chosen_i], 4),
            "combined_before_feedback": round(combined[chosen_i], 4),
            "feedback_adjustment": round(fb_adj[chosen_i], 4),
            "final_score": round(score, 4),
            "threshold": threshold,
            "matched_keywords": _matched_keywords(text, entry),
        }
        if skipped_avoided:
            explanation["session_avoid_skipped"] = deeplinks[ranked_indices[0]]
        if entry.deeplink in avoid_deeplinks:
            explanation["already_tried_this_session"] = True
        if score < threshold:
            explanation["rejected_reason"] = "final_score below threshold"
            return None, score, explanation
        return entry, score, explanation


class RulesDeeplinkIndex:
    """Variant A (fallback): deterministic fuzzy keyword matching."""

    def __init__(self, entries: list[DeeplinkEntry]):
        self.entries = [e for e in entries if e.deeplink != DUMMY_POSITIVE_DEEPLINK]

    def best_match(self, text: str, threshold: float = 45.0) -> tuple[DeeplinkEntry | None, float]:
        best_entry, best_score = None, 0.0
        for e in self.entries:
            score = fuzz.token_set_ratio(text, e.corpus_text) + 100 * feedback.get_adjustment(e.deeplink)
            if score > best_score:
                best_score, best_entry = score, e
        if best_entry is None or best_score < threshold:
            return None, best_score / 100.0
        return best_entry, best_score / 100.0

    def search(self, text: str, top_k: int = 3):
        scored = [
            (e, max(0.0, fuzz.token_set_ratio(text, e.corpus_text) / 100.0
                    + feedback.get_adjustment(e.deeplink)))
            for e in self.entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def best_match_explained(
        self, text: str, threshold: float = 45.0, avoid_deeplinks=None
    ) -> tuple[DeeplinkEntry | None, float, dict]:
        """Rules-variant counterpart to HybridDeeplinkIndex.best_match_explained()
        -- same idea (return the winning entry's score components), but for
        the single fuzzy-ratio score this matcher uses instead of a
        BM25+dense blend. `avoid_deeplinks` (Task 36) works identically to
        the hybrid variant's -- see _pick_index_avoiding()."""
        if not self.entries:
            return None, 0.0, {"matcher": "rules_fuzzy", "reason": "empty_index"}
        avoid_deeplinks = avoid_deeplinks or frozenset()

        fuzzy_scores = [fuzz.token_set_ratio(text, e.corpus_text) for e in self.entries]
        fb_pts = [100 * feedback.get_adjustment(e.deeplink) for e in self.entries]
        scores = [f + p for f, p in zip(fuzzy_scores, fb_pts)]

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Original implementation only ever set best_entry on a strictly-
        # positive score (best_score started at 0.0, compared with ">") --
        # preserve that: an all-zero-or-negative field is still "no match".
        if scores[ranked_indices[0]] <= 0.0:
            return None, 0.0, {"matcher": "rules_fuzzy", "reason": "no_entry_scored_above_zero"}

        deeplinks = [e.deeplink for e in self.entries]
        chosen_i, skipped_avoided = _pick_index_avoiding(
            ranked_indices, scores, deeplinks, threshold, avoid_deeplinks
        )
        entry, best_score = self.entries[chosen_i], scores[chosen_i]

        explanation = {
            "matcher": "rules_fuzzy",
            "fuzzy_score": round(fuzzy_scores[chosen_i], 2),
            "feedback_adjustment_pts": round(fb_pts[chosen_i], 2),
            "final_score_pts": round(best_score, 2),
            "threshold_pts": threshold,
            "matched_keywords": _matched_keywords(text, entry),
        }
        if skipped_avoided:
            explanation["session_avoid_skipped"] = deeplinks[ranked_indices[0]]
        if entry.deeplink in avoid_deeplinks:
            explanation["already_tried_this_session"] = True
        if best_score < threshold:
            explanation["rejected_reason"] = "final_score below threshold"
            return None, best_score / 100.0, explanation
        return entry, best_score / 100.0, explanation


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
                              path: str = "deeplinks.json",
                              avoid_deeplinks=None) -> tuple[dict, dict | None]:
    """Used by pipeline.py's Stage 2. Returns (actionableDeeplink dict,
    validationDeeplink dict | None). `avoid_deeplinks` (Task 36) is an
    optional set of deeplinks this session already tried and marked
    unhelpful -- see session_memory.py."""
    index = get_index(variant, path)
    query_text = action_name + " " + " ".join(steps)
    entry, score, explanation = index.best_match_explained(query_text, avoid_deeplinks=avoid_deeplinks)

    if entry is not None:
        actionable = {
            "deeplink": entry.deeplink,
            "description": entry.description,
            "message": entry.message,
            "originalType": entry.original_type,
            "matchExplanation": explanation,
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
        "matchExplanation": explanation,
    }, None
