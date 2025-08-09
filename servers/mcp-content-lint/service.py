# lint_service.py
# Async, deterministic content-lint service with YAML-driven rules.
# Backward-compatible outputs; extended numeric-claim checks, H2-as-questions,
# and optional readability-by-page-type targeting.

import os
import re
import yaml
import math
import pathlib
from typing import Any, Dict, List, Optional, Tuple

# ---------- Config / Loading ----------

_DEFAULT_RULES = pathlib.Path(__file__).parent / "rules" / "default.yaml"
_DEFAULT_NAP_FILE = pathlib.Path(__file__).parents[2] / "packages" / "local" / "nap_truth.yaml"
_DEFAULT_NAP_KEY = os.getenv("DEFAULT_LOCAL_KEY", os.getenv("DEFAULT_GEO_KEY", "wilmington_nc"))

_RULES_CACHE: Optional[Dict[str, Any]] = None
_NAP_CACHE: Optional[Dict[str, Any]] = None


def load_rules() -> Dict[str, Any]:
    """Load YAML rules from CONTENT_LINT_RULES or default path; cache result."""
    global _RULES_CACHE
    if _RULES_CACHE is not None:
        return _RULES_CACHE
    rules_path = pathlib.Path(os.getenv("CONTENT_LINT_RULES", str(_DEFAULT_RULES)))
    if not rules_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    with rules_path.open("r", encoding="utf-8") as f:
        _RULES_CACHE = yaml.safe_load(f) or {}
    # Baselines / fallbacks
    r = _RULES_CACHE
    r.setdefault("requirements", {})
    r["requirements"].setdefault("tldr", {"markers": ["TL;DR:", "Summary:", "TLDR:"], "window_chars": 400})
    r["requirements"].setdefault(
        "comparison", {"triggers": [" vs ", "comparison", "compare", "pros and cons"], "table_required": True}
    )
    r["requirements"].setdefault(
        "citations",
        {
            "require_when_numbers": True,
            "number_regex": r"(?:\u0024\d[\d,]*(?:\.\d+)?|\b\d{4}\b|\b\d+(?:%|x)\b)",
            "allowed_citation_patterns": [r"\[[0-9]+\]", r"(https?://[^ )\]]+)"],
        },
    )
    # NEW: numeric-claim window + allowed units
    r.setdefault(
        "numeric_claims",
        {"require_citation": True, "window_sentences": 3, "allowed_units": ["ft", "%", "$", "in", "mph", "psi"]},
    )
    r.setdefault("phrases", {}).setdefault("forbidden", [])
    r.setdefault("brand_glossary", {"canonical": "", "variants": []})
    r.setdefault("readability", {"target_grade": 8, "long_sentence_threshold": 28, "very_long_sentence_threshold": 40})
    r.setdefault("severity", {})
    r.setdefault("ordering", {"sort_by": ["span.start", "rule", "message"]})
    r.setdefault("quotables", {"min_count": 5, "max_len_chars": 140})
    return r


def load_nap_truth() -> Dict[str, Any]:
    """Load local market/NAP truth YAML (locations keyed); cache full file."""
    global _NAP_CACHE
    if _NAP_CACHE is not None:
        return _NAP_CACHE
    nap_env = os.getenv("LOCAL_NAP_FILE") or os.getenv("GEO_NAP_FILE") or str(_DEFAULT_NAP_FILE)
    nap_file = pathlib.Path(nap_env)
    if not nap_file.exists():
        _NAP_CACHE = {"locations": {}}
        return _NAP_CACHE
    with nap_file.open("r", encoding="utf-8") as f:
        _NAP_CACHE = yaml.safe_load(f) or {"locations": {}}
    _NAP_CACHE.setdefault("locations", {})
    return _NAP_CACHE


def _severity_for(rule: str) -> str:
    return load_rules().get("severity", {}).get(rule, "warn")


# ---------- Span Mapping Utilities ----------

def _line_starts(text: str) -> List[int]:
    starts = [0]
    for m in re.finditer(r"\n", text):
        starts.append(m.end())
    return starts


def _offset_to_line_span(text: str, start: int, end: int) -> Dict[str, int]:
    """Map absolute offsets to { line (1-based), start, end } relative to line start."""
    if start < 0:
        start = 0
    if end < start:
        end = start
    line_starts = _line_starts(text)
    # binary search
    lo, hi, line_idx = 0, len(line_starts) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if line_starts[mid] <= start:
            line_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    line_start_offset = line_starts[line_idx]
    return {"line": line_idx + 1, "start": start - line_start_offset, "end": end - line_start_offset}


def _find_spans_ci(text: str, needle: str) -> List[Tuple[int, int]]:
    """Case-insensitive substring matches -> list[(start, end)]."""
    out: List[Tuple[int, int]] = []
    if not needle:
        return out
    tl, nl = text.lower(), needle.lower()
    i = tl.find(nl)
    while i != -1:
        out.append((i, i + len(nl)))
        i = tl.find(nl, i + 1)
    return out


# ---------- Sentence segmentation with offsets ----------

_SENT_RE = re.compile(r"[^.!?]+[.!?]", re.MULTILINE | re.DOTALL)

def _sentences(text: str) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, sentence_text)."""
    sents: List[Tuple[int, int, str]] = []
    for m in _SENT_RE.finditer(text):
        sents.append((m.start(), m.end(), m.group(0)))
    return sents


# ---------- Readability (FKGL) ----------

def _syllable_count(word: str) -> int:
    w = re.sub(r"[^A-Za-z]", "", word).lower()
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    syl = len(groups)
    if w.endswith("e") and syl > 1:
        syl -= 1
    return max(1, syl)


def fkgl(text: str) -> float:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if not words:
        return 0.0
    syllables = sum(_syllable_count(w) for w in words)
    s_count = max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))
    w_count = len(words)
    grade = 0.39 * (w_count / s_count) + 11.8 * (syllables / w_count) - 15.59
    return round(grade, 2)


# ---------- Style Lints ----------

def _has_tldr(text: str, rules: Dict[str, Any]) -> bool:
    tldr = rules["requirements"]["tldr"]
    window = int(tldr.get("window_chars", 400))
    markers = [m.lower() for m in tldr.get("markers", [])]
    head = text[:window].lower()
    return any(m in head for m in markers)


def _detect_table_presence(text: str) -> bool:
    # HTML table
    if re.search(r"<table\b", text, re.IGNORECASE):
        return True
    # GFM table (header + --- separator)
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        if "|" in lines[i] and re.search(r"^\s*\|?\s*:?-{3,}\s*(\|\s*:?-{3,}\s*)+\|?\s*$", lines[i + 1]):
            return True
    # Fallback heuristic
    if "|" in text and re.search(r"^-{3,}", text, re.MULTILINE):
        return True
    return False


def _quotables(text: str, rules: Dict[str, Any]) -> Tuple[int, List[Tuple[int, int]]]:
    max_len = int(rules.get("quotables", {}).get("max_len_chars", 140))
    count = 0
    spans: List[Tuple[int, int]] = []
    for m in _SENT_RE.finditer(text):
        s = m.group(0).strip()
        if len(s) <= max_len:
            count += 1
            spans.append((m.start(), m.end()))
    return count, spans


def lint_style(text: str) -> Dict[str, Any]:
    rules = load_rules()
    issues: List[Dict[str, Any]] = []

    # TL;DR
    if not _has_tldr(text, rules):
        rule = "tldr.missing"
        issues.append(
            {"rule": rule, "severity": _severity_for(rule), "span": {"line": 1, "start": 0, "end": 0},
             "fix": "Add a TL;DR within the first 400 characters, e.g., 'TL;DR: …'"} )

    # Forbidden phrases
    for item in rules.get("phrases", {}).get("forbidden", []):
        phr = item.get("phrase", "")
        if not phr:
            continue
        for (a, b) in _find_spans_ci(text, phr):
            span = _offset_to_line_span(text, a, b)
            rule = "forbidden.phrase"
            issues.append(
                {"rule": rule, "severity": _severity_for(rule), "span": span,
                 "fix": item.get("suggestion") or f"Replace '{phr}' with a concrete description."} )

    # Comparison pages require a table
    cmp_req = rules["requirements"]["comparison"]
    triggers = [x.lower() for x in cmp_req.get("triggers", [])]
    lower = text.lower()
    if any(t in lower for t in triggers) and not _detect_table_presence(text):
        pos = -1; trig = ""
        for t in triggers:
            pos = lower.find(t)
            if pos != -1:
                trig = t; break
        span = _offset_to_line_span(text, pos if pos != -1 else 0, (pos + len(trig)) if pos != -1 else 0)
        rule = "comparison.table_missing"
        issues.append(
            {"rule": rule, "severity": _severity_for(rule), "span": span,
             "fix": "Add a comparison table (Markdown or HTML) summarizing key attributes."} )

    # Brand glossary
    brand = rules.get("brand_glossary", {})
    canonical = brand.get("canonical", "")
    for v in brand.get("variants", []):
        pattern = v.get("pattern", "")
        if not pattern:
            continue
        for (a, b) in _find_spans_ci(text, pattern):
            span = _offset_to_line_span(text, a, b)
            rule = "brand.glossary.mismatch"
            fix_to = v.get("fix") or canonical or pattern
            issues.append(
                {"rule": rule, "severity": _severity_for(rule), "span": span,
                 "fix": f"Use the brand's canonical form: '{fix_to}'."} )

    # Quotables
    min_q = int(rules.get("quotables", {}).get("min_count", 5))
    q_count, _ = _quotables(text, rules)
    if q_count < min_q:
        rule = "quotables.too_few"
        issues.append(
            {"rule": rule, "severity": _severity_for(rule), "span": {"line": 1, "start": 0, "end": 0},
             "fix": f"Add at least {min_q} short, standalone sentences (≤ {rules['quotables']['max_len_chars']} chars)."} )

    # NEW: H2 as questions (scannability)
    # Find Markdown H2 lines starting with "## " that don't end with '?'
    for m in re.finditer(r"(?m)^(##\s+.+)$", text):
        hline = m.group(1).strip()
        if not hline.endswith("?"):
            span = _offset_to_line_span(text, m.start(1), m.end(1))
            rule = "scannability.h2_question"
            issues.append(
                {"rule": rule, "severity": _severity_for(rule), "span": span,
                 "fix": "Phrase H2 headings as questions to improve scannability (end with '?')."} )

    # Deterministic ordering
    def _key(i: Dict[str, Any]) -> Tuple[int, int, str, str]:
        s = i.get("span") or {}
        return (int(s.get("line", 0)), int(s.get("start", 0)), str(i.get("rule", "")), str(i.get("fix", "")))
    issues.sort(key=_key)
    return {"issues": issues}


# ---------- Readability Lint ----------

def lint_readability(text: str, page_type: Optional[str] = None, target_fkgl_by_type: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    rules = load_rules()
    target = int(rules["readability"]["target_grade"])
    if page_type and isinstance(target_fkgl_by_type, dict):
        try:
            t = int(target_fkgl_by_type.get(page_type))  # type: ignore
            if t > 0:
                target = t
        except Exception:
            pass

    g = fkgl(text)
    notes: List[str] = []
    long_thresh = int(rules["readability"]["long_sentence_threshold"])
    very_long_thresh = int(rules["readability"]["very_long_sentence_threshold"])
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    long_count = sum(1 for s in sentences if len(s.split()) > long_thresh)
    very_long_count = sum(1 for s in sentences if len(s.split()) > very_long_thresh)
    if g > target:
        notes.append(f"Aim for grade ≤ {target}.")
    if long_count:
        notes.append(f"{long_count} sentences exceed {long_thresh} words; consider splitting.")
    if very_long_count:
        notes.append(f"{very_long_count} sentences exceed {very_long_thresh} words; split aggressively.")
    return {"grade": g, "target": target, "notes": notes}


# ---------- Citations / Numeric-claim Lint ----------

def lint_citations(text: str, window_sentences: Optional[int] = None) -> Dict[str, Any]:
    rules = load_rules()
    cit = rules["requirements"]["citations"]
    num_cfg = rules.get("numeric_claims", {})
    require_numbers = bool(cit.get("require_when_numbers", True))
    window = int(window_sentences if window_sentences is not None else num_cfg.get("window_sentences", 3))

    num_re = re.compile(cit["number_regex"])
    allowed_list = cit.get("allowed_citation_patterns", [])
    allowed_re = re.compile("|".join(f"(?:{p})" for p in allowed_list)) if allowed_list else re.compile(r"^$")

    # Simple unit match (best-effort) for reporting; not strictly required
    allowed_units = set(str(u).lower() for u in (num_cfg.get("allowed_units") or []))
    unit_re = re.compile(r"(?P<num>\$?\d[\d,]*(?:\.\d+)?)(?:\s*(?P<unit>[A-Za-z%$]+))?")

    sents = _sentences(text)
    total_claims = 0
    supported_claims = 0
    issues: List[Dict[str, Any]] = []

    # For each sentence with a numeric token, check for citation in window
    for idx, (a, b, sent) in enumerate(sents):
        nums = list(num_re.finditer(sent))
        if not nums:
            continue
        needs = require_numbers
        # Search for allowed citation in [idx-window, idx+window]
        has_cite = False
        lo = max(0, idx - window)
        hi = min(len(sents) - 1, idx + window)
        for j in range(lo, hi + 1):
            if allowed_re.search(sents[j][2]):
                has_cite = True
                break

        for m in nums:
            total_claims += 1
            if has_cite or not needs:
                supported_claims += 1
                continue
            # Flag this numeric claim
            gstart = a + m.start()
            gend = a + m.end()
            span = _offset_to_line_span(text, gstart, gend)
            # Optional unit analysis
            um = unit_re.match(m.group(0))
            unit = (um.group("unit").lower() if um and um.group("unit") else None)
            unit_note = ""
            if unit and allowed_units and unit not in allowed_units:
                unit_note = f" (unit '{unit}' not in allowed units)"
            issues.append(
                {"rule": "citations.numeric_claim_missing",
                 "severity": _severity_for("citations.numeric_claim_missing"),
                 "span": span,
                 "reason": f"Numeric claim '{m.group(0)}' lacks a citation within ±{window} sentences{unit_note}.",
                 "suggestion": "Add a footnote [n] or a source URL near the claim."}
            )

    # Legacy booleans for backward compat
    requires_any = bool(num_re.search(text)) if require_numbers else False
    has_any_citation = bool(allowed_re.search(text))
    suggestions: List[str] = []
    if requires_any and not has_any_citation and not issues:
        # Fallback suggestion (legacy shape)
        for val in num_re.findall(text):
            suggestions.append(f"Add a citation for '{val}' (e.g., footnote [1] or a source URL).")

    score = (supported_claims / total_claims) if total_claims else 1.0
    return {"has_citation": has_any_citation, "suggestions": suggestions, "issues": issues, "score": round(score, 3)}


# ---------- Local Market / NAP Lint ----------

def _normalize_phone(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\D+", "", s)


def lint_geo(text: str, geo_key: Optional[str] = None) -> Dict[str, Any]:
    """Validate NAP presence against local market/NAP truth set (geo_key optional)."""
    nap_all = load_nap_truth()
    loc_key = geo_key or _DEFAULT_NAP_KEY
    nap = (nap_all.get("locations") or {}).get(loc_key) or {}

    issues: List[Dict[str, Any]] = []
    tel_truth = _normalize_phone(nap.get("telephone"))
    city_truth = (((nap.get("address") or {}).get("addressLocality")) or "").strip()
    region_truth = (((nap.get("address") or {}).get("addressRegion")) or "").strip()

    if tel_truth:
        found = _normalize_phone(text)
        if tel_truth not in found:
            issues.append(
                {"rule": "geo.nap_mismatch", "severity": _severity_for("geo.nap_mismatch"),
                 "span": {"line": 1, "start": 0, "end": 0},
                 "fix": "Ensure canonical telephone is present in normalized form."}
            )

    if city_truth and city_truth.lower() not in text.lower():
        issues.append(
            {"rule": "geo.city_presence", "severity": _severity_for("geo.city_presence"),
             "span": {"line": 1, "start": 0, "end": 0},
             "fix": f"Include city name '{city_truth}' in body."}
        )

    if region_truth and region_truth.lower() not in text.lower():
        issues.append(
            {"rule": "geo.region_presence", "severity": _severity_for("geo.region_presence"),
             "span": {"line": 1, "start": 0, "end": 0},
             "fix": f"Include region/State '{region_truth}' where relevant."}
        )

    return {"issues": issues}


# ---------- JSON-RPC Handler ----------

async def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Methods:
      - lint.style        { text } -> { issues: [...] }
      - lint.readability  { text, page_type?, target_fkgl_by_type? } -> { grade, target, notes[] }
      - lint.citations    { text, window_sentences? } -> { has_citation, suggestions[], issues[], score }
      - lint.geo          { text, geo_key? } -> { issues: [...] }
    """
    mid = req.get("id")
    method = req.get("method")
    p = req.get("params") or {}

    def ok(result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code: str, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        e: Dict[str, Any] = {"code": code, "message": message}
        if data:
            e["data"] = data
        return {"jsonrpc": "2.0", "id": mid, "error": e}

    if method == "lint.style":
        text = p.get("text")
        if not isinstance(text, str):
            return err("invalid_params", "text must be a string")
        return ok(lint_style(text))

    if method == "lint.readability":
        text = p.get("text")
        if not isinstance(text, str):
            return err("invalid_params", "text must be a string")
        page_type = p.get("page_type")
        target_map = p.get("target_fkgl_by_type")
        return ok(lint_readability(text, page_type=page_type, target_fkgl_by_type=target_map))

    if method == "lint.citations":
        text = p.get("text")
        if not isinstance(text, str):
            return err("invalid_params", "text must be a string")
        window = p.get("window_sentences")
        try:
            window_i = int(window) if window is not None else None
        except Exception:
            return err("invalid_params", "window_sentences must be an integer")
        return ok(lint_citations(text, window_sentences=window_i))

    if method == "lint.geo":
        text = p.get("text")
        if not isinstance(text, str):
            return err("invalid_params", "text must be a string")
        geo_key = p.get("geo_key")
        return ok(lint_geo(text, geo_key=geo_key))

    return err("method_not_found", str(method))
