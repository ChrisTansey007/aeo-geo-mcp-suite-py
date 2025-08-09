# search_service.py
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

API_VERSION = "1.1"

# ------------------------
# Helpers & configuration
# ------------------------

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def _endswith_host(host: str, site: str) -> bool:
    """
    True if host belongs to site. Accepts bare domain or wildcard like *.example.com.
    Enforces label boundary to avoid 'notexample.com' false positives.
    """
    if not site:
        return True
    site = site.lstrip("*.").lower()
    host = (host or "").lower()
    return host == site or host.endswith("." + site)

def _parse_host(url: str) -> str:
    # lightweight hostname extractor
    m = re.match(r"^[a-z]+://([^/]+)", url or "", re.I)
    return (m.group(1) if m else "").split("@")[-1].split(":")[0]

def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")

def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

def _parse_iso_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # Support YYYY-MM-DD or full ISO8601
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return datetime.fromisoformat(s)
        # Normalize Z
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None else default

# Hybrid config (env-overridable)
HYBRID_MODE = _env("HYBRID_MODE", "rrf")  # rrf | weighted
HYBRID_RRF_K = int(_env("HYBRID_RRF_K", "60"))
HYBRID_NORMALIZE = _env("HYBRID_NORMALIZE", "zscore")  # zscore|minmax|none
HYBRID_BM25_WEIGHT = float(_env("HYBRID_BM25_WEIGHT", "0.5"))
HYBRID_VEC_WEIGHT = float(_env("HYBRID_VEC_WEIGHT", "0.5"))

# Vector backend (hashing trick; no external ANN)
VEC_BACKEND = _env("VEC_BACKEND", "hash")  # hash | none
VECTOR_DIM = int(_env("VECTOR_DIM", "512"))

@dataclass(frozen=True)
class Doc:
    id: str
    url: str
    title: str
    text: str
    anchor: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None  # may include last_modified, place_ids, jurisdiction


# ------------------------
# In-memory index state
# ------------------------

class _Index:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.corpus_version = 0
        self.docs: List[Doc] = []
        self.ids_by_pos: List[str] = []
        self.pos_by_id: Dict[str, int] = {}
        self.bm25: Optional[BM25Okapi] = None
        # vector space (hashing)
        self.vec_dim: int = VECTOR_DIM
        self.doc_vecs: Optional[np.ndarray] = None  # shape [N, D], L2-normalized

    # --- vector helpers (hashing trick) ---
    def _hashing_vec(self, text: str) -> np.ndarray:
        """Project tokens into a fixed-dim bag via hashing; L2-normalize."""
        v = np.zeros(self.vec_dim, dtype=np.float32)
        for t in _tokenize(text):
            h = int(_sha1_hex(t), 16) % self.vec_dim
            v[h] += 1.0
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def _rebuild(self) -> None:
        self.ids_by_pos = [d.id for d in self.docs]
        self.pos_by_id = {d.id: i for i, d in enumerate(self.docs)}
        corpus_tokens = [_tokenize(f"{d.title} {d.text}") for d in self.docs]
        self.bm25 = BM25Okapi(corpus_tokens) if self.docs else None

        # Build vector space if enabled
        if VEC_BACKEND == "hash" and self.docs:
            mat = np.zeros((len(self.docs), self.vec_dim), dtype=np.float32)
            for i, d in enumerate(self.docs):
                mat[i, :] = self._hashing_vec(f"{d.title} {d.text}")
            self.doc_vecs = mat
        else:
            self.doc_vecs = None

        self.corpus_version += 1

    def build(self, docs: List[Dict[str, Any]]) -> int:
        with self._lock:
            self.docs = [
                Doc(
                    id=d["id"],
                    url=d.get("url", ""),
                    title=d.get("title", ""),
                    text=d.get("text", ""),
                    anchor=d.get("anchor"),
                    meta=d.get("meta"),
                )
                for d in docs
            ]
            self._rebuild()
            return len(self.docs)

    def upsert(self, doc: Dict[str, Any]) -> None:
        """Simple upsert (rebuilds indices for correctness)."""
        with self._lock:
            idx = self.pos_by_id.get(doc["id"])
            new_doc = Doc(
                id=doc["id"],
                url=doc.get("url", ""),
                title=doc.get("title", ""),
                text=doc.get("text", ""),
                anchor=doc.get("anchor"),
                meta=doc.get("meta"),
            )
            if idx is None:
                self.docs.append(new_doc)
            else:
                self.docs[idx] = new_doc
            self._rebuild()

    def clear(self) -> None:
        with self._lock:
            self.docs = []
            self.ids_by_pos = []
            self.pos_by_id = {}
            self.bm25 = None
            self.doc_vecs = None
            self.corpus_version += 1

    # --- query helpers ---
    def bm25_search(self, q: str, top_k: int) -> List[Tuple[str, float]]:
        with self._lock:
            if not self.bm25 or not self.docs:
                return []
            toks = _tokenize(q)
            if not toks:
                return []
            scores = self.bm25.get_scores(toks)
            pairs = sorted(
                ((self.ids_by_pos[i], float(scores[i])) for i in range(len(scores))),
                key=lambda x: (-x[1], x[0]),
            )
            return pairs[:max(0, top_k)]

    def vector_search(self, q: str, top_k: int) -> List[Tuple[str, float]]:
        with self._lock:
            if VEC_BACKEND == "none" or self.doc_vecs is None or not self.docs:
                return []
            qv = self._hashing_vec(q)
            # cosine via dot product (already L2-normalized)
            scores = self.doc_vecs @ qv  # shape [N]
            idxs = np.argpartition(scores, -min(len(scores), top_k * 6))[-min(len(scores), top_k * 6):]
            pairs = [(self.ids_by_pos[i], float(scores[i])) for i in idxs]
            pairs.sort(key=lambda x: (-x[1], x[0]))
            return pairs[:max(0, top_k)]

    def get_doc(self, doc_id: str) -> Optional[Doc]:
        with self._lock:
            pos = self.pos_by_id.get(doc_id)
            return self.docs[pos] if pos is not None else None

_INDEX = _Index()


# ------------------------
# Snippets, highlighting & boosting
# ------------------------

def _highlight(s: str, terms: List[str]) -> str:
    """Wrap query terms with <em>..</em>, case-insensitive, non-overlapping."""
    if not s or not terms:
        return s
    uniq = sorted(set(terms), key=len, reverse=True)
    pattern = r"(" + "|".join(re.escape(t) for t in uniq) + r")"
    try:
        return re.sub(pattern, lambda m: f"<em>{m.group(0)}</em>", s, flags=re.IGNORECASE)
    except re.error:
        return s

def _snippet(text: str, q: str, max_len: int = 220) -> str:
    if not text:
        return ""
    ql = q.lower()
    tl = text.lower()
    hit = tl.find(ql)
    if hit == -1:
        terms = [t for t in _tokenize(q) if t]
        positions = [tl.find(t) for t in terms if tl.find(t) != -1]
        hit = min(positions) if positions else -1
    if hit == -1:
        head = text[:max_len]
        return head + ("…" if len(text) > len(head) else "")
    start = max(0, hit - max_len // 3)
    end = min(len(text), hit + (2 * max_len) // 3)
    snip = ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")
    return _highlight(snip, _tokenize(q))

def _geo_aliases() -> Dict[str, List[str]]:
        """
        Load local market aliases once. Format expectation (example):
            markets:
                wilmington_nc:
                    city: ["Wilmington"]
                    county: ["New Hanover"]
                    neighbors: ["Wrightsville Beach", "Carolina Beach"]
        Env:
            LOCAL_ALIASES_PATH (preferred)
            GEO_ALIASES_PATH (fallback)
            LOCAL_MARKET (preferred)
            GEO_MARKET (fallback)
        """
        path = (
                os.getenv("LOCAL_ALIASES_PATH")
                or os.getenv("GEO_ALIASES_PATH")  # backward-compat
                or os.path.join("packages", "local", "aliases.yaml")
        )
        market = os.getenv("LOCAL_MARKET") or os.getenv("GEO_MARKET") or "wilmington_nc"
        try:
                import yaml, pathlib
                p = pathlib.Path(path)
                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                return data.get("markets", {}).get(market, {}) or {}
        except Exception:
                return {}

_GEO = _geo_aliases()

def _geo_boost(base_score: float, title: str, query: str) -> float:
    boost = 0.0
    try:
        cities = (_GEO.get("city", []) or []) + (_GEO.get("neighbors", []) or []) + (_GEO.get("county", []) or [])
        hay = f"{title} {query}".lower()
        for name in cities:
            n = (name or "").lower().strip()
            if not n:
                continue
            if re.search(rf"\b{re.escape(n)}\b", hay):
                boost += 2.0
    except Exception:
        pass
    return base_score + boost


# ------------------------
# Fusion & normalization
# ------------------------

def _rrf(lists: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for lst in lists:
        for i, (doc_id, _score) in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

def _normalize(scores: Dict[str, float], method: str) -> Dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float64)
    if method == "minmax":
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi - lo <= 1e-12:
            return {k: 0.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
    if method == "zscore":
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        if sd <= 1e-12:
            return {k: 0.0 for k in scores}
        return {k: (v - mu) / sd for k, v in scores.items()}
    return dict(scores)

def _weighted_fuse(bm: List[Tuple[str, float]], vec: List[Tuple[str, float]], norm: str, wb: float, wv: float) -> List[Tuple[str, float]]:
    bm_map = {i: s for i, s in bm}
    vc_map = {i: s for i, s in vec}
    ids = set(bm_map) | set(vc_map)
    bm_n = _normalize(bm_map, norm)
    vc_n = _normalize(vc_map, norm)
    fused = [(i, wb * bm_n.get(i, 0.0) + wv * vc_n.get(i, 0.0)) for i in ids]
    fused.sort(key=lambda x: (-x[1], x[0]))
    return fused

# ------------------------
# Pagination tokens (deterministic)
# ------------------------

def _make_page_token(q: str, offset: int, corpus_version: int) -> str:
    payload = {"q": _sha1_hex(q)[:8], "o": int(offset), "v": int(corpus_version)}
    return _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

def _parse_page_token(token: Optional[str], q: str, corpus_version: int) -> int:
    if not token:
        return 0
    try:
        obj = json.loads(_b64url_decode(token))
        if obj.get("q") != _sha1_hex(q)[:8] or int(obj.get("v", -1)) != int(corpus_version):
            return 0
        off = int(obj.get("o", 0))
        return off if off >= 0 else 0
    except Exception:
        return 0


# ------------------------
# Filters
# ------------------------

def _doc_passes_filters(doc: Doc, filters: Optional[Dict[str, Any]], legacy_site: Optional[str]) -> bool:
    if not doc:
        return False
    if not filters and not legacy_site:
        return True

    # domain / site
    host = _parse_host(doc.url)
    if legacy_site and not _endswith_host(host, legacy_site):
        return False
    if filters:
        domains = filters.get("domain_whitelist") or []
        if domains:
            ok = any(_endswith_host(host, d) for d in domains if isinstance(d, str))
            if not ok:
                return False

        # lastmod_gte
        lastmod_gte = filters.get("lastmod_gte")
        if lastmod_gte:
            gate = _parse_iso_date(lastmod_gte)
            doc_last = _parse_iso_date((doc.meta or {}).get("last_modified") if doc.meta else None)
            if gate and doc_last and doc_last < gate:
                return False

        # jurisdiction
        j = filters.get("jurisdiction")
        if j:
            j = str(j).lower()
            d_j = str((doc.meta or {}).get("jurisdiction", "")).lower()
            if d_j and d_j != j:
                return False

        # place_ids overlap
        want_places = filters.get("place_ids") or []
        if want_places:
            doc_places = (doc.meta or {}).get("place_ids") or []
            if isinstance(doc_places, list):
                doc_set = set(str(dp).lower() for dp in doc_places)
                if not any(str(p).lower() in doc_set for p in want_places):
                    return False

    return True


# ------------------------
# Public API
# ------------------------

def seed_corpus_if_empty() -> None:
    if _INDEX.bm25 is not None or _INDEX.docs:
        return
    _INDEX.build(
        [
            {
                "id": "doc1",
                "url": "https://example.com/",
                "title": "Port City Fence – Wilmington NC",
                "text": "Vinyl fence care and pricing for Wilmington NC.",
                "anchor": None,
                "meta": {"last_modified": "2025-01-01"},
            },
            {
                "id": "doc2",
                "url": "https://example.com/vinyl",
                "title": "Vinyl vs Wood – Pros and Cons",
                "text": "Comparison table and costs. Wilmington and Wrightsville Beach examples.",
                "anchor": "b-0004-aaaa",
                "meta": {"last_modified": "2025-02-01", "place_ids": ["place-wilmington-nc"]},
            },
        ]
    )

def _validate_top_k(v: Any, default: int = 10) -> int:
    try:
        n = int(v)
        return max(1, min(100, n))
    except Exception:
        return default

def _result_from_doc(
    doc: Doc,
    score: float,
    include_snippets: bool,
    q: str,
    explain: bool = False,
    bm25_score: Optional[float] = None,
    vector_score: Optional[float] = None,
    fusion: Optional[str] = None,
) -> Dict[str, Any]:
    anchored = doc.url + (("#" + doc.anchor) if doc.anchor else "")
    res: Dict[str, Any] = {
        "url": anchored,                 # back-compat (includes anchor)
        "anchored_url": anchored,        # explicit
        "title": doc.title,
        "score": round(float(score), 6),
        "source": "local-hybrid" if vector_score is not None else "local-bm25",
    }
    if include_snippets:
        res["snippet"] = _snippet(doc.text, q)
    if explain:
        res["bm25_score"] = round(float(bm25_score if bm25_score is not None else score), 6)
        res["vector_score"] = (round(float(vector_score), 6) if isinstance(vector_score, (int, float)) else None)
        if fusion:
            res["fusion"] = fusion
    return res


async def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Methods:
      search.health {} -> { api_version, corpus_version, hybrid: {mode, normalize, rrf_k, bm25_weight, vec_weight, backend} }

      search.query  { q: string, site?: string, top_k?: int, include_snippets?: bool, page_token?: string,
                      filters?: { lastmod_gte?: string, place_ids?: string[], jurisdiction?: string, domain_whitelist?: string[] },
                      debug?: bool }
        -> { results: [{url,anchored_url,title,snippet?,score,source}], next_page_token?: string, debug?: {...} }

      search.hybrid { q: string, filters?: object, top_k?: int, page_token?: string, include_snippets?: bool, explain?: bool, debug?: bool }
        -> { results: [...], next_page_token?: string, debug?: {...} }

      search.corpus.build { docs: [{id,url,title,text,anchor?,meta?}] } -> { count: number, version: number }
      search.corpus.upsert { doc: {id,url,title,text,anchor?,meta?} } -> { ok: true, version: number }
      search.corpus.clear  {} -> { ok: true, version: number }
    """
    mid = req.get("id")
    method = req.get("method")
    p = req.get("params", {}) or {}

    # Ensure default seed exists for out-of-the-box behavior
    seed_corpus_if_empty()

    try:
        if method == "search.health":
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "api_version": API_VERSION,
                    "corpus_version": _INDEX.corpus_version,
                    "hybrid": {
                        "mode": HYBRID_MODE,
                        "normalize": HYBRID_NORMALIZE,
                        "rrf_k": HYBRID_RRF_K,
                        "bm25_weight": HYBRID_BM25_WEIGHT,
                        "vec_weight": HYBRID_VEC_WEIGHT,
                        "backend": VEC_BACKEND,
                    },
                },
            }

        if method == "search.corpus.build":
            docs = p.get("docs") or []
            if not isinstance(docs, list) or not all(isinstance(d, dict) and "id" in d for d in docs):
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": "invalid_params", "message": "docs[] with 'id' required"}}
            cnt = _INDEX.build(docs)
            return {"jsonrpc": "2.0", "id": mid, "result": {"count": cnt, "version": _INDEX.corpus_version}}

        if method == "search.corpus.upsert":
            doc = p.get("doc") or {}
            if not isinstance(doc, dict) or "id" not in doc:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": "invalid_params", "message": "doc with 'id' required"}}
            _INDEX.upsert(doc)
            return {"jsonrpc": "2.0", "id": mid, "result": {"ok": True, "version": _INDEX.corpus_version}}

        if method == "search.corpus.clear":
            _INDEX.clear()
            return {"jsonrpc": "2.0", "id": mid, "result": {"ok": True, "version": _INDEX.corpus_version}}

        if method == "search.query":
            t0 = time.perf_counter()
            q = p.get("q")
            if not isinstance(q, str) or not q.strip():
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": "invalid_params", "message": "q (string) required"}}
            site = p.get("site")
            top_k = _validate_top_k(p.get("top_k"), 10)
            include_snippets = bool(p.get("include_snippets", True))
            page_token = p.get("page_token")
            filters = p.get("filters") or {}
            debug = bool(p.get("debug", False))

            # compute base ranking (overfetch to absorb filters)
            base = _INDEX.bm25_search(q, top_k=top_k * 6)
            # geo boost + deterministic tie-break
            boosted: List[Tuple[str, float]] = []
            for doc_id, score in base:
                d = _INDEX.get_doc(doc_id)
                if not d:
                    continue
                s = _geo_boost(score, d.title, q)
                boosted.append((doc_id, s))
            boosted.sort(key=lambda x: (-x[1], x[0]))

            # site+filters + pagination
            filtered: List[Tuple[str, float]] = []
            for doc_id, s in boosted:
                d = _INDEX.get_doc(doc_id)
                if not d:
                    continue
                if not _doc_passes_filters(d, filters, site):
                    continue
                filtered.append((doc_id, s))

            offset = _parse_page_token(page_token, q, _INDEX.corpus_version)
            window = filtered[offset : offset + top_k]
            next_token = None
            if offset + top_k < len(filtered):
                next_token = _make_page_token(q, offset + top_k, _INDEX.corpus_version)

            results = []
            for doc_id, s in window:
                d = _INDEX.get_doc(doc_id)
                if d:
                    results.append(_result_from_doc(d, s, include_snippets, q))

            out: Dict[str, Any] = {"results": results}
            if next_token:
                out["next_page_token"] = next_token
            if debug:
                out["debug"] = {"latency_ms": round((time.perf_counter() - t0) * 1000, 2), "corpus_version": _INDEX.corpus_version}
            return {"jsonrpc": "2.0", "id": mid, "result": out}

        if method == "search.hybrid":
            t0 = time.perf_counter()
            q = p.get("q")
            if not isinstance(q, str) or not q.strip():
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": "invalid_params", "message": "q (string) required"}}
            top_k = _validate_top_k(p.get("top_k"), 10)
            page_token = p.get("page_token")
            include_snippets = bool(p.get("include_snippets", True))
            explain = bool(p.get("explain", False))
            filters = p.get("filters") or {}
            debug = bool(p.get("debug", False))

            bm_list = _INDEX.bm25_search(q, top_k=top_k * 6)
            vec_list = _INDEX.vector_search(q, top_k=top_k * 6)

            if HYBRID_MODE == "weighted":
                fused = _weighted_fuse(bm_list, vec_list, HYBRID_NORMALIZE, HYBRID_BM25_WEIGHT, HYBRID_VEC_WEIGHT)
                fusion_label = "weighted"
            else:
                # default RRF; if no vectors, this is just BM25 in RRF clothes
                lists = [bm_list] + ([vec_list] if vec_list else [])
                fused = _rrf(lists, k=HYBRID_RRF_K)
                fusion_label = "rrf"

            # Apply filters
            filtered: List[Tuple[str, float]] = []
            for doc_id, fused_score in fused:
                d = _INDEX.get_doc(doc_id)
                if not d:
                    continue
                if not _doc_passes_filters(d, filters, None):
                    continue
                filtered.append((doc_id, fused_score))

            offset = _parse_page_token(page_token, q, _INDEX.corpus_version)
            window = filtered[offset : offset + top_k]
            next_token = None
            if offset + top_k < len(filtered):
                next_token = _make_page_token(q, offset + top_k, _INDEX.corpus_version)

            results = []
            bm_map = {doc_id: score for doc_id, score in bm_list}
            vc_map = {doc_id: score for doc_id, score in vec_list}
            for doc_id, fused_score in window:
                d = _INDEX.get_doc(doc_id)
                if not d:
                    continue
                results.append(
                    _result_from_doc(
                        d,
                        fused_score,
                        include_snippets,
                        q,
                        explain=explain,
                        bm25_score=bm_map.get(doc_id),
                        vector_score=vc_map.get(doc_id),
                        fusion=fusion_label if explain else None,
                    )
                )

            out: Dict[str, Any] = {"results": results}
            if next_token:
                out["next_page_token"] = next_token
            if debug:
                out["debug"] = {"latency_ms": round((time.perf_counter() - t0) * 1000, 2), "corpus_version": _INDEX.corpus_version}
            return {"jsonrpc": "2.0", "id": mid, "result": out}

        return {"jsonrpc": "2.0", "id": mid, "error": {"code": "method_not_found", "message": str(method)}}

    except Exception as e:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": "internal", "message": str(e)}}  # keep payload minimal
