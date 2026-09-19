"""
offline_fallback.py — "the fallback tech-stack": a fully deterministic,
zero-API-key replacement for Stage 0 (query enrichment) and Stage 1
(structure extraction) that pipeline.py switches to automatically when the
LLM path is unavailable (no LLM_API_KEY configured) or fails (network
error, rate limit, malformed JSON after retries).

Why this exists: the primary pipeline (prompts.py + llm_client.py) gives
the best quality but depends on an external API key and network access —
a real risk on demo day or in a judge's offline environment. This module
gets the service to a fully-working, schema-compliant state with $0 cost
and no network calls at all, by doing rule-based extraction straight from
the provided siis_response text (which also has the nice side effect of
zero hallucination risk, since nothing here can invent a step that isn't
in the source text).

Nothing in this file calls an LLM. Deeplink matching is NOT duplicated
here — it reuses deeplink_matching.py, the exact same module the LLM path
uses for Stage 2.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

# ---------------------------------------------------------------------
# Query enrichment (offline replacement for Stage 0 / prompts.STAGE0_*)
# ---------------------------------------------------------------------

_DEVICE_PATTERN = re.compile(
    r"\b(samsung|galaxy|note\s?\d*|z\s?fold\s?\d*|z\s?flip\s?\d*|"
    r"[as]\d{1,3}(\s?(ultra|plus|fe|\+))?|tab\s?[a-z]?\d*)\b",
    re.IGNORECASE,
)
_CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "doesn't": "does not",
    "don't": "do not", "isn't": "is not", "aren't": "are not",
    "didn't": "did not", "i'm": "i am", "it's": "it is",
    "wasn't": "was not", "hasn't": "has not", "haven't": "have not",
    "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
}
_FILLER_LEAD = re.compile(r"^\s*(my|the|i\s+have|i've\s+got|i\s+got)\s+", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "the", "my", "is", "are", "was", "were", "i", "it", "and",
    "or", "to", "of", "in", "on", "for", "with", "so", "this", "that",
    "after", "when", "while", "not", "but", "have", "has", "had", "be",
    "been", "being", "just", "even", "also", "no", "any", "other",
}


def _expand_contractions(text: str) -> str:
    out = text
    for k, v in _CONTRACTIONS.items():
        out = re.sub(re.escape(k), v, out, flags=re.IGNORECASE)
    return out


def normalize_query(raw_query: str) -> str:
    q = re.sub(r"^\s*\d+[\.\)]\s*", "", raw_query.strip())
    q = _expand_contractions(q).replace("*", "")
    q = re.sub(r"\s+", " ", q).strip().rstrip(".! ")
    return q


def core_problem_phrase(raw_query: str) -> str:
    q = _DEVICE_PATTERN.sub("", normalize_query(raw_query))
    q = _FILLER_LEAD.sub("", q)
    return re.sub(r"\s+", " ", q).strip(" ,") or normalize_query(raw_query)


def significant_tokens(text: str) -> set[str]:
    text = text.translate(str.maketrans("", "", string.punctuation))
    return {t.lower() for t in text.split() if len(t) > 2} - _STOPWORDS


def _inject_typo(text: str) -> str:
    for a, b in (("tion", "toin"), ("screen", "screne"), ("ing", "ign")):
        if a in text:
            return text.replace(a, b, 1)
    words = text.split()
    if not words:
        return text
    longest = max(words, key=len)
    idx = words.index(longest)
    for i, ch in enumerate(longest):
        if i not in (0, len(longest) - 1) and ch.lower() in "aeiou":
            words[idx] = longest[:i] + longest[i + 1:]
            break
    return " ".join(words)


def generate_paraphrases(raw_query: str, n: int = 10) -> list[str]:
    """8-10 deterministic paraphrases spanning formal/casual/keyword-only/
    frustrated/typo-inclusive registers -- same contract as the LLM
    prompt's Stage 0 output, just template-generated instead of modeled."""
    core = core_problem_phrase(raw_query)
    core_lower = core[:1].lower() + core[1:] if core else core
    canon = normalize_query(raw_query)

    candidates = [
        canon,
        f"Why does my phone have {core_lower}?",
        f"My phone has {core_lower}.",
        core,
        f"phone {core_lower}",
        f"ugh my phone has {core_lower}, so annoying",
        f"pls help, {core_lower}",
        f"{core_lower} pls help fix",
        _inject_typo(f"my phone has {core_lower}"),
        " ".join(sorted(significant_tokens(core))),
    ]
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip()
        key = c.lower()
        if c and key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) >= n:
            break
    return out[:n]


def offline_enrich(raw_complaint: str) -> dict:
    """Drop-in replacement for stage0_enrich()'s return shape."""
    return {
        "technical_query": normalize_query(raw_complaint),
        "query_variations": generate_paraphrases(raw_complaint),
    }


# ---------------------------------------------------------------------
# Structure extraction (offline replacement for Stage 1 / STAGE1_*)
# ---------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,3})\s*(.+?)\s*$", re.MULTILINE)

_IMPERATIVE_VERBS = (
    "go", "open", "tap", "swipe", "navigate", "select", "enable", "disable",
    "turn", "press", "hold", "check", "restart", "reboot", "update",
    "clear", "toggle", "choose", "touch", "power", "remove", "back",
    "contact", "visit", "schedule", "connect", "install", "uninstall",
    "reset", "close", "drag", "adjust", "scroll", "plug", "unplug",
    "charge", "wipe", "reinsert", "insert", "download", "sign", "log",
    "switch", "disconnect", "reconnect", "force", "hard", "try",
    "ensure", "make", "verify", "confirm", "allow", "grant", "set",
)

# Transitional leads that some articles put BEFORE the actual imperative
# ("Then, try to turn it on...", "Next, tap Settings...") -- stripped
# before checking whether the remainder is an instruction, so the step
# text itself doesn't carry the throwaway "Then," prefix.
_TRANSITIONAL_LEAD_RE = re.compile(
    r"^(then|next|after that|finally|now|once (?:that's )?done)\s*,?\s*",
    re.IGNORECASE,
)
_NON_STEP_LEADS = (
    "note:", "if ", "some ", "available ", "this ", "not all", "with some",
    "once ", "when you", "you can", "you will", "you'll", "your device",
    "remember", "before ", "however", "for instance", "keep in mind",
    "important:", "tip:",
)
_CRITICAL_KEYWORDS = (
    "factory reset", "firmware update", "restart your", "reboot your",
    "safe mode", "wipe", "erase all data", "reset the",
)
_MANUAL_KEYWORDS = (
    "service center", "visit a samsung", "schedule a repair",
    "contact samsung", "physical", "hardware", "replace",
)


@dataclass
class Section:
    header: str
    body: str


def _split_sentences(body: str) -> list[str]:
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", body) if p.strip()]


def _is_step_sentence(sentence: str) -> bool:
    low = sentence.lower().strip()
    if any(low.startswith(lead) for lead in _NON_STEP_LEADS):
        return False
    low_stripped = _TRANSITIONAL_LEAD_RE.sub("", low)
    first_word = re.split(r"[\s,]", low_stripped, maxsplit=1)[0] if low_stripped else ""
    if first_word in _IMPERATIVE_VERBS:
        return True
    for clause in re.split(r",\s*(?:and\s+)?then\s+|,\s*and\s+", sentence):
        cw = clause.strip().lower().split(" ", 1)
        if cw and cw[0] in _IMPERATIVE_VERBS:
            return True
    return False


def parse_sections(content: str) -> list[Section]:
    matches = list(_HEADER_RE.finditer(content))
    if not matches:
        sentences = _split_sentences(content)
        return [Section(header=(sentences[0][:60] if sentences else "General"), body=content)]
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append(Section(header=m.group(2).strip(), body=content[start:end]))
    return sections


def extract_steps(section: Section) -> list[str]:
    steps = [s for s in _split_sentences(section.body) if _is_step_sentence(s)]
    cleaned = []
    for s in steps:
        s = re.sub(r"^(and\s+then\s+|then\s+)", "", s, flags=re.IGNORECASE)
        s = _TRANSITIONAL_LEAD_RE.sub("", s)
        cleaned.append((s[0].upper() + s[1:]) if s else s)
    return cleaned


def classify_category(header: str, body: str) -> str:
    text = (header + " " + body).lower()
    if any(k in text for k in _CRITICAL_KEYWORDS):
        return "critical"
    if any(k in text for k in _MANUAL_KEYWORDS):
        return "manual"
    return "auto"


def section_relevance(query_core: str, section: Section) -> float:
    q_tokens = significant_tokens(query_core)
    s_tokens = significant_tokens(section.header + " " + section.body)
    if not q_tokens or not s_tokens:
        return 0.0
    return len(q_tokens & s_tokens) / len(q_tokens)


# ---------------------------------------------------------------------
# Contract phrasing helpers (mirrors validators.py's rules)
# ---------------------------------------------------------------------

_TITLE_STOPWORDS = {
    "my", "the", "a", "an", "and", "or", "is", "are", "was", "were", "goes",
    "went", "go", "completely", "suddenly", "no", "not", "so", "it", "on",
    "in", "at", "with", "after", "when", "while", "to", "of", "for",
    "any", "other", "too", "just", "even", "some", "about", "than",
    "also", "keeps", "keep", "very", "really", "still",
}
_TITLE_DOMAIN_PRIORITY = [
    "screen", "display", "battery", "touch", "touchscreen", "gesture",
    "gestures", "navigation", "swipe", "camera", "flicker", "flickers",
    "crack", "cracked", "blank", "black", "charge", "charging", "speaker",
    "microphone", "bluetooth", "wifi", "volume", "brightness",
    "notification", "notifications", "fingerprint", "app", "apps",
    "performance", "storage", "update", "rotate", "rotation", "delay",
    "lag", "laggy", "email", "sync",
]


def to_sentence_case(text: str) -> str:
    text = text.strip()
    return (text[0].upper() + text[1:]) if text else text


def to_title_case(text: str) -> str:
    small = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "or"}
    words = text.strip().split()
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        out.append(lw if (i > 0 and lw in small) else lw[:1].upper() + lw[1:])
    return " ".join(out)


def make_title(topic_phrase: str, max_words: int = 3) -> str:
    raw_words = re.findall(r"[A-Za-z0-9']+", topic_phrase)
    content_words = [w for w in raw_words if w.lower() not in _TITLE_STOPWORDS] or raw_words
    priority = [w for w in content_words if w.lower() in _TITLE_DOMAIN_PRIORITY]
    rest = [w for w in content_words if w not in priority]
    chosen: list[str] = []
    for w in priority + rest:
        if w.lower() not in {c.lower() for c in chosen}:
            chosen.append(w)
        if len(chosen) >= max_words:
            break
    if len(chosen) < 2:
        chosen = (chosen + ["device", "settings"])[:2]
    return to_sentence_case(" ".join(chosen[:max_words]))


def make_goal(title: str) -> str:
    return f"Follow these steps to perform this {to_title_case(title)} Troubleshooting"


_BENEFIT_STOPWORDS = {"the", "a", "an", "to", "for", "of", "your"}


def make_description(benefit_phrase: str) -> str:
    """validators.check_description_format() counts the WHOLE string
    ('It will' included) and requires 5-7 words total, matching the
    canonical worked example in Appendix B ('It will let you choose
    navigation type' = 7 words total) -- so content after 'It will' must
    be 3 to 5 words, not 5 to 7."""
    words = [w for w in re.findall(r"[A-Za-z0-9']+", benefit_phrase.lower())
             if w not in _BENEFIT_STOPWORDS] or re.findall(r"[A-Za-z0-9']+", benefit_phrase.lower())
    if len(words) < 3:
        for p in ("your", "device", "settings", "quickly"):
            if len(words) >= 3:
                break
            if p not in words:
                words.append(p)
    return "It will " + " ".join(words[:5])


# ---------------------------------------------------------------------
# Orchestration: offline replacement for stage1_extract()
# ---------------------------------------------------------------------

NO_MATCH_SECTION_THRESHOLD = 0.12
_MANUAL_LEAD_KEYWORDS = ("visit a samsung", "service center", "contact samsung",
                          "schedule a repair", "walk-in service")
_CRITICAL_LEAD_KEYWORDS = (
    "factory reset", "restart", "reboot", "safe mode", "wipe", "erase",
    "power off", "turn off your phone", "force restart",
)


def _step_category(step: str, section_category: str) -> str:
    """Category is judged from the STEP itself, not inherited wholesale
    from its section -- a section titled 'Restart Your Phone in Safe
    Mode' still contains several harmless navigation steps ('swipe down
    to open Quick settings') that shouldn't be dragged to 'critical' just
    because they're neighbours of the actual disruptive step."""
    low = step.lower()
    if any(k in low for k in _CRITICAL_LEAD_KEYWORDS):
        return "critical"
    if any(k in low for k in _MANUAL_LEAD_KEYWORDS):
        return "manual"
    return "auto" if section_category == "critical" else section_category


def _looks_like_ui_action(step: str) -> bool:
    low = step.lower()
    return any(h in low for h in ("settings", "tap", "toggle", "enable", "disable",
                                   "turn on", "turn off", "select", "navigate",
                                   "swipe", "screen", "menu"))


def _build_single_goal(core: str, siis_response: str) -> dict | None:
    """Core single-issue extraction: given ONE problem phrase and the
    reference text, returns one Goal dict or None (no viable match)."""
    sections = parse_sections(siis_response)
    scored = [(sec, section_relevance(core, sec)) for sec in sections]
    best_section_relevance = max((s for _, s in scored), default=0.0)
    whole_doc_relevance = section_relevance(core, Section(header="", body=siis_response))
    best_relevance = max(best_section_relevance, whole_doc_relevance)

    if best_relevance < NO_MATCH_SECTION_THRESHOLD:
        return None

    relevant_sections = [sec for sec, score in scored
                          if score >= max(NO_MATCH_SECTION_THRESHOLD, best_section_relevance * 0.5)]
    if not relevant_sections:
        relevant_sections = sections

    steps_with_ctx: list[tuple[str, str]] = []
    seen_steps: set[str] = set()
    for sec in relevant_sections:
        cat = classify_category(sec.header, sec.body)
        for step in extract_steps(sec):
            key = step.strip().lower()
            if key in seen_steps:
                continue
            seen_steps.add(key)
            steps_with_ctx.append((step, cat))

    if not steps_with_ctx:
        return None

    # Group into one Action per distinct target: since deeplink matching
    # happens later in pipeline.py (shared Stage 2, same as the LLM path),
    # here we just group consecutive same-category steps loosely by simple
    # keyword clustering so each Action stays screen-sized.
    actions = _group_steps(steps_with_ctx)
    if not actions:
        return None

    title = make_title(core)
    score = round(min(0.99, 0.55 + 0.45 * best_relevance), 2)
    return {
        "goal": make_goal(title),
        "title": title,
        "score": score,
        "actions": actions,
    }


def offline_extract(technical_query: str, siis_response: str, core_problem: str | None = None) -> dict:
    """Drop-in replacement for stage1_extract()'s return shape:
    {"contexts": [Goal-shaped-dict, ...]}  (empty list = no_match).

    Handles compound complaints ("battery dies fast and camera lags") by
    splitting into independent sub-issues (see split_multi_issue) and
    building a separate Goal per sub-issue that clears the relevance gate
    against the SAME reference text -- so a single request can legitimately
    come back with 2+ contexts, exactly like the schema's `contexts: List[Goal]`
    was designed to allow. Falls back to the original single-issue
    behaviour whenever no confident split is found, so this is fully
    backward compatible with every already-passing official query.
    """
    if not siis_response or not siis_response.strip():
        return {"contexts": []}

    sub_queries = split_multi_issue(core_problem or technical_query)
    if len(sub_queries) == 1:
        core = core_problem or core_problem_phrase(technical_query)
        goal = _build_single_goal(core, siis_response)
        return {"contexts": [goal] if goal else []}

    contexts = []
    for sub_q in sub_queries:
        sub_core = core_problem_phrase(sub_q)
        goal = _build_single_goal(sub_core, siis_response)
        if goal is not None:
            contexts.append(goal)

    if not contexts:
        # None of the sub-issues found grounding -- try once more treating
        # the complaint as a single issue before giving up entirely.
        core = core_problem or core_problem_phrase(technical_query)
        goal = _build_single_goal(core, siis_response)
        return {"contexts": [goal] if goal else []}

    contexts.sort(key=lambda g: g["score"], reverse=True)
    return {"contexts": contexts}


# --- Multi-issue splitting -------------------------------------------

_DOMAIN_GROUPS = {
    "battery": {"battery", "charge", "charging", "drain"},
    "camera": {"camera", "photo", "video", "flash", "lens"},
    "screen": {"screen", "display", "touch", "touchscreen", "gesture",
               "swipe", "flicker", "crack", "blank", "black", "brightness",
               "rotation", "rotate"},
    "performance": {"slow", "lag", "laggy", "performance", "storage",
                     "freeze", "crash", "update", "app"},
    "connectivity": {"wifi", "bluetooth", "network", "signal", "data",
                      "hotspot"},
    "audio": {"speaker", "microphone", "volume", "sound", "mic", "audio"},
}

# Only "," and "and" are treated as genuine clause separators. A bare
# "also"/"as well" is deliberately NOT a split point on its own -- it very
# often sits mid-clause ("camera also lags"), not between two clauses, and
# splitting there would fragment a single sentence.
_SPLIT_RE = re.compile(r"\s*,\s*(?:and\s+)?|\s+\band\b\s+", re.IGNORECASE)


def _stem(token: str) -> str:
    """Crude suffix stripping so 'lags'/'draining'/'crashes' match the
    same domain keyword as 'lag'/'drain'/'crash' without a full stemmer
    dependency."""
    for suffix in ("ing", "es", "ed", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _dominant_domains(tokens: set[str]) -> set[str]:
    stems = {_stem(t) for t in tokens} | tokens
    hits = set()
    for domain, kws in _DOMAIN_GROUPS.items():
        kw_stems = {_stem(k) for k in kws} | kws
        if stems & kw_stems:
            hits.add(domain)
    return hits


def split_multi_issue(query_text: str) -> list[str]:
    """Splits a compound complaint into independent sub-complaints ONLY
    when it clearly names 2+ domain-disjoint issues (e.g. "battery dies
    fast and camera lags on open"). Returns [query_text] unchanged
    otherwise -- deliberately conservative, since a false split would
    silently fragment a single coherent complaint (worse than occasionally
    missing a real multi-issue case)."""
    parts = [p.strip(" .") for p in _SPLIT_RE.split(query_text) if p.strip(" .")]
    if len(parts) < 2:
        return [query_text]

    part_domains = [_dominant_domains(significant_tokens(p)) for p in parts]
    if any(not d for d in part_domains):
        return [query_text]  # a part with no recognizable domain -> too risky

    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if part_domains[i] & part_domains[j]:
                return [query_text]  # shared domain -> one issue described in detail

    return parts


def _group_steps(steps_with_ctx: list[tuple[str, str]]) -> list[dict]:
    """Cluster steps into screen-sized Actions keyed by which catalog
    deeplink they resolve to -- i.e. the one-action-one-screen rule is
    enforced by "do these steps land on the same real Settings screen?",
    not by shallow word overlap between step sentences. Steps that don't
    resolve to any catalog entry become their own manual/no-deeplink
    groups. This reuses the exact same matcher the LLM path's Stage 2
    uses, just called per-step here instead of per-already-grouped-action.
    """
    from deeplink_matching import get_index, to_title_case as dm_title_case, DUMMY_POSITIVE_DEEPLINK

    index = get_index("hybrid")
    groups: dict[str, dict] = {}
    order: list[str] = []

    for step, section_category in steps_with_ctx:
        category = _step_category(step, section_category)

        if category == "manual" and not _looks_like_ui_action(step):
            group_key = "manual:no-deeplink"
            entry, explanation = None, None
        else:
            entry, _score, explanation = index.best_match_explained(step)
            group_key = entry.id if entry is not None else f"dummy:{step[:24].lower()}"

        if group_key not in groups:
            order.append(group_key)
            action_name_source = re.sub(
                r"^(tap|select|go to|navigate to|open|enable|disable|swipe|press|hold)\s+",
                "", step, flags=re.IGNORECASE,
            )
            fallback_name = dm_title_case(re.sub(r"[^A-Za-z0-9 ]", "", action_name_source)[:30]) or "Device Settings"
            groups[group_key] = {
                "steps": [],
                "category": category,
                "entry": entry,
                "explanation": explanation,
                "action_name": (dm_title_case(entry.message) if (entry and entry.message) else fallback_name),
            }
        else:
            rank = {"auto": 0, "manual": 1, "critical": 2}
            if rank.get(category, 1) > rank.get(groups[group_key]["category"], 1):
                groups[group_key]["category"] = category
        groups[group_key]["steps"].append(step)

    rank = {"auto": 0, "manual": 1, "critical": 2}
    ordered_keys = sorted(order, key=lambda k: rank.get(groups[k]["category"], 1))

    actions = []
    for key in ordered_keys:
        g = groups[key]
        entry = g["entry"]
        if entry is not None:
            actionable = {
                "deeplink": entry.deeplink,
                "description": entry.description,
                "message": entry.message,
                "originalType": entry.original_type,
                "matchExplanation": g["explanation"],
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
            description = make_description(entry.qna_description or entry.description)
        elif key.startswith("dummy:") and g["category"] != "manual":
            actionable = {
                "deeplink": "bixby://dummy_positive",
                "description": f"Opens the {g['action_name'].lower()} settings screen on the device.",
                "message": f"Open {g['action_name']}",
                "originalType": "placeholder",
                "matchExplanation": g["explanation"],
            }
            validation = None
            description = make_description(g["action_name"])
        else:
            actionable = None
            validation = None
            description = make_description(g["action_name"])

        actions.append({
            "actionName": g["action_name"],
            "description": description,
            "category": g["category"],
            "stepGroups": [{
                "steps": g["steps"],
                "actionableDeeplink": actionable,
                "validationDeeplink": validation,
            }],
        })
    return actions
