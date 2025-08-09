# crawl_service.py
# Robust async crawler + canonicalizer with robots, ETag/IMS, retries, stable block IDs.
# PDF table-first extraction (Camelot/Tabula), pluggable HTML cleaners, and a simple recrawl cache.

import asyncio
import base64
import datetime as dt
import hashlib
import json
import os
import random
import re
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from pdfminer_high_level import extract_text_to_fp as _pdfminer_extract_text_to_fp  # type: ignore
try:
    # Some envs have pdfminer under pdfminer.high_level; gracefully handle both
    from pdfminer.high_level import extract_text_to_fp as _pdfminer_extract_text_to_fp  # type: ignore
except Exception:
    pass

# Optional libs – used if available
try:
    import camelot  # type: ignore
    _HAS_CAMELOT = True
except Exception:
    _HAS_CAMELOT = False

try:
    import tabula  # type: ignore
    _HAS_TABULA = True
except Exception:
    _HAS_TABULA = False

try:
    import trafilatura  # type: ignore
    _HAS_TRAFILATURA = True
except Exception:
    _HAS_TRAFILATURA = False

try:
    import justext  # type: ignore
    _HAS_JUSTEXT = True
except Exception:
    _HAS_JUSTEXT = False

# If you have a shared helper, keep it; otherwise fallback to internal
try:
    from packages.mcp_common.anchors import stable_block_id  # type: ignore
except Exception:
    def stable_block_id(xpath: str, payload: str, order: int) -> str:
        h = hashlib.sha1((xpath + "|" + payload).encode("utf-8")).hexdigest()[:4]
        return f"b-{order:04d}-{h}"

USER_AGENT = os.getenv("CRAWL_UA", "AEO-GEO-MCP-Crawler/1.2")
MAX_BYTES = int(os.getenv("CRAWL_MAX_BYTES", "8388608"))  # 8 MiB cap
RETRY_MAX = int(os.getenv("CRAWL_RETRIES", "2"))
ROBOT_TTL = int(os.getenv("CRAWL_ROBOTS_TTL", "600"))     # seconds

# In-memory robots cache: { netloc: (expires_ts, RobotRules) }
_ROBOTS: Dict[str, Tuple[float, "RobotRules"]] = {}

# Simple recrawl cache: { url: {etag,last_modified,attempts,backoff_until} }
_CACHE: Dict[str, Dict[str, Any]] = {}


class RobotRules:
    """Tiny robots.txt allow/disallow store for our UA (prefix-based disallow)."""
    def __init__(self, allow_all: bool = True, disallow: Optional[List[str]] = None):
        self.allow_all = allow_all
        self.disallow = disallow or []

    def allowed(self, path: str) -> bool:
        if self.allow_all:
            return True
        lp = path or "/"
        for rule in self.disallow:
            if rule and lp.startswith(rule):
                return False
        return True


async def _fetch_robots(client: httpx.AsyncClient, url: str) -> RobotRules:
    parsed = urlparse(url)
    netloc = parsed.netloc
    now = asyncio.get_event_loop().time()
    cached = _ROBOTS.get(netloc)
    if cached and cached[0] > now:
        return cached[1]

    rules = RobotRules(allow_all=True)
    robots_url = f"{parsed.scheme}://{netloc}/robots.txt"
    try:
        r = await client.get(robots_url, timeout=5.0, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200 and "text" in (r.headers.get("content-type") or ""):
            txt = r.text
            ua_blocks: Dict[str, Dict[str, List[str]]] = {}
            current_ua: Optional[str] = None
            for line in txt.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"(?i)(user-agent)\s*:\s*(.+)", line)
                if m:
                    current_ua = m.group(2).strip()
                    ua_blocks.setdefault(current_ua.lower(), {"allow": [], "disallow": []})
                    continue
                if current_ua:
                    for key in ("allow", "disallow"):
                        m2 = re.match(rf"(?i)({key})\s*:\s*(.+)", line)
                        if m2:
                            ua_blocks[current_ua.lower()][key].append(m2.group(2).strip())
            block = ua_blocks.get(USER_AGENT.lower()) or ua_blocks.get("*")
            if block:
                rules = RobotRules(allow_all=(len(block["disallow"]) == 0), disallow=block["disallow"])
    except Exception:
        pass

    _ROBOTS[netloc] = (now + ROBOT_TTL, rules)
    return rules


async def _get_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        follow_redirects=True,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
        },
    )


async def _backoff_sleep(attempt: int):
    base = 0.5 * (2 ** attempt)
    await asyncio.sleep(base + random.uniform(0, 0.25))


def _cache_get_validators(url: str, ims: Optional[str], etag: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Prefer explicit IMS/ETag; else use cached validators if present."""
    if ims or etag:
        return ims, etag
    row = _CACHE.get(url) or {}
    return row.get("last_modified"), row.get("etag")


def _cache_update_after(url: str, meta: Dict[str, Any], ok: bool, status: int):
    row = _CACHE.setdefault(url, {})
    row["etag"] = meta.get("etag") or row.get("etag")
    row["last_modified"] = meta.get("last_modified") or row.get("last_modified")
    row["last_status"] = status
    now = asyncio.get_event_loop().time()
    if ok:
        row["attempts"] = 0
        row["backoff_until"] = 0
        row["last_success_at"] = now
    else:
        attempts = int(row.get("attempts") or 0) + 1
        row["attempts"] = attempts
        # basic exponential backoff (capped at 60s)
        row["backoff_until"] = now + min(60.0, (2 ** attempts))
    _CACHE[url] = row


async def _fetch(
    url: str,
    timeout_ms: int = 15000,
    ims: Optional[str] = None,
    etag: Optional[str] = None,
    respect_robots: bool = True,
) -> Tuple[Optional[bytes], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Returns (content_bytes or None, metadata, error or None).
    Adds simple recrawl cache with validators and backoff.
    """
    # backoff check
    row = _CACHE.get(url)
    if row and float(row.get("backoff_until") or 0) > asyncio.get_event_loop().time():
        return None, {"final_url": url}, {"code": "backing_off", "message": "Retry later due to prior failures"}

    headers: Dict[str, str] = {}
    ims, etag = _cache_get_validators(url, ims, etag)
    if ims:
        headers["If-Modified-Since"] = ims
    if etag:
        headers["If-None-Match"] = etag

    async with await _get_client() as client:
        # robots allow check
        if respect_robots:
            rules = await _fetch_robots(client, url)
            parsed = urlparse(url)
            if not rules.allowed(parsed.path or "/"):
                return None, {}, {"code": "robots_disallowed", "message": "Blocked by robots.txt"}

        last_exc: Optional[Exception] = None
        for attempt in range(RETRY_MAX + 1):
            try:
                r = await client.get(url, headers=headers, timeout=timeout_ms / 1000)
                meta = {
                    "status": r.status_code,
                    "final_url": str(r.url),
                    "mime": (r.headers.get("content-type") or "application/octet-stream").split(";")[0].strip(),
                    "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
                    "etag": r.headers.get("etag"),
                    "last_modified": r.headers.get("last-modified"),
                    "content_length": int(r.headers.get("content-length") or 0),
                }
                if r.status_code == 304:
                    _cache_update_after(url, meta, ok=True, status=304)
                    return None, meta, {"code": "not_modified", "message": "ETag/IMS unchanged"}

                content = r.content
                if len(content) > MAX_BYTES:
                    _cache_update_after(url, meta, ok=False, status=r.status_code)
                    return None, meta, {"code": "content_too_large", "message": f"Response exceeded {MAX_BYTES} bytes"}

                r.raise_for_status()
                _cache_update_after(url, meta, ok=True, status=r.status_code)
                return content, meta, None

            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else 0
                if status in (429, 500, 502, 503, 504) and attempt < RETRY_MAX:
                    await _backoff_sleep(attempt)
                    continue
                _cache_update_after(url, {"etag": etag, "last_modified": ims}, ok=False, status=status)
                return None, {"status": status, "final_url": url}, {"code": "fetch_failed", "message": f"HTTP {status}"}
            except Exception as e:
                last_exc = e
                if attempt < RETRY_MAX:
                    await _backoff_sleep(attempt)
                    continue
                _cache_update_after(url, {"etag": etag, "last_modified": ims}, ok=False, status=0)
                return None, {"final_url": url}, {"code": "fetch_exception", "message": str(last_exc)}

    return None, {"final_url": url}, {"code": "internal", "message": "unexpected"}


def _extract_jsonld(html: str, base_url: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            raw = tag.string or tag.text or ""
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    out.append({"jsonld": item, "source_url": base_url})
            else:
                out.append({"jsonld": data, "source_url": base_url})
        except Exception:
            continue
    return out


def _meta_extracts(soup: BeautifulSoup) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    link_canon = soup.find("link", attrs={"rel": lambda v: v and "canonical" in (v if isinstance(v, list) else [v])})
    if link_canon and link_canon.get("href"):
        meta["canonical"] = link_canon["href"]

    robots = soup.find("meta", attrs={"name": re.compile(r"robots", re.I)})
    robostr = (robots.get("content") or "").lower() if robots else ""
    if robostr:
        meta["robots"] = {"noindex": "noindex" in robostr, "nofollow": "nofollow" in robostr}

    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        meta["lang"] = html_tag["lang"]

    nap: Dict[str, Any] = {}
    tel = soup.select_one('a[href^="tel:"]')
    if tel:
        nap["telephone"] = (tel.get("href", "") or "").replace("tel:", "")
    addr = soup.find(attrs={"itemtype": re.compile("PostalAddress", re.I)}) or soup.find("address")
    if addr:
        nap["address_text"] = addr.get_text(" ", strip=True)

    geo: Dict[str, Any] = {}
    maps = soup.select_one('a[href*="goo.gl/maps"], a[href*="google.com/maps"]')
    if maps and maps.get("href"):
        geo["hasMap"] = maps["href"]

    if nap:
        meta["nap"] = nap
    if geo:
        meta["geo"] = geo
    return meta


def _xpath_for(el: Tag) -> str:
    parts: List[str] = []
    cur: Optional[Tag] = el
    while cur is not None and isinstance(cur, Tag):
        name = cur.name
        idx = 1
        sib = cur.previous_sibling
        while sib is not None:
            if isinstance(sib, Tag) and sib.name == name:
                idx += 1
            sib = sib.previous_sibling
        parts.append(f"{name}[{idx}]")
        cur = cur.parent if isinstance(cur.parent, Tag) else None
    parts.reverse()
    return "/" + "/".join(parts)


def _collect_table(tbl: Tag) -> List[List[str]]:
    rows: List[List[str]] = []
    thead = tbl.find("thead")
    if thead:
        for tr in thead.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if any(cell for cell in cells):
                rows.append(cells)
    for tr in tbl.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if any(cell for cell in cells):
            if not rows or cells != rows[-1]:
                rows.append(cells)
    return rows


def _canon_html(html: str, base_url: str, cleaner_engine: str = "bs4", keep_toc: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Cleaner selection (best-effort). We always parse with bs4 for structure/tables.
    soup = BeautifulSoup(html, "lxml")

    # Optionally reduce boilerplate with trafilatura/justext text extraction,
    # but still keep original soup for tables/lists.
    main_text_only: Optional[str] = None
    if cleaner_engine == "trafilatura" and _HAS_TRAFILATURA:
        try:
            main_text_only = trafilatura.extract(html) or None
        except Exception:
            main_text_only = None
    elif cleaner_engine == "justext" and _HAS_JUSTEXT:
        try:
            paragraphs = justext.justext(html, justext.get_stoplist("English"))
            main_text_only = "\n\n".join(p.text for p in paragraphs if not p.is_boilerplate) or None
        except Exception:
            main_text_only = None

    # Basic sanitization (drop script/style/noscript)
    for s in soup(["script", "style", "noscript"]):
        s.decompose()

    # If keep_toc is False, drop common TOC/nav
    if not keep_toc:
        for nav in soup.select("[role=navigation], nav, [id*=toc], [class*=toc]"):
            nav.decompose()

    blocks: List[Dict[str, Any]] = []
    order = 0

    # If we obtained "main text only", turn it into paragraph blocks first
    if main_text_only:
        for para in [p.strip() for p in re.split(r"\n{2,}", main_text_only) if p.strip()]:
            order += 1
            # synthetic xpath for extracted text
            xpath = f"/extracted/p[{order}]"
            bid = stable_block_id(xpath, para, order)
            blocks.append({"id": bid, "type": "para", "text": para, "xpath": xpath})

    # Then scan the HTML structure for headings/lists/tables (to preserve them)
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table"]):
        if re.match(r"h[1-6]", el.name or ""):
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            order += 1
            xpath = _xpath_for(el)
            bid = stable_block_id(xpath, text, order)
            blocks.append({"id": bid, "type": "heading", "text": text, "xpath": xpath})
            continue

        if el.name == "p":
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            order += 1
            xpath = _xpath_for(el)
            bid = stable_block_id(xpath, text, order)
            blocks.append({"id": bid, "type": "para", "text": text, "xpath": xpath})
            continue

        if el.name in ("ul", "ol"):
            items = [li.get_text(" ", strip=True) for li in el.find_all("li", recursive=False)]
            items = [i for i in items if i]
            if not items:
                continue
            joined = "\n".join(items)
            order += 1
            xpath = _xpath_for(el)
            bid = stable_block_id(xpath, joined, order)
            blocks.append({"id": bid, "type": "list", "text": joined, "items": items, "xpath": xpath})
            continue

        if el.name == "table":
            rows = _collect_table(el)
            if not rows:
                continue
            payload = json.dumps(rows, ensure_ascii=False)
            order += 1
            xpath = _xpath_for(el)
            bid = stable_block_id(xpath, payload, order)
            blocks.append({"id": bid, "type": "table", "cells": rows, "xpath": xpath})
            continue

    meta = _meta_extracts(soup)
    return blocks, meta


def _pdf_to_text(content: bytes) -> str:
    out = BytesIO()
    try:
        _pdfminer_extract_text_to_fp(BytesIO(content), out)  # type: ignore
        return out.getvalue().decode("utf-8", "ignore")
    except Exception:
        return ""


def _pdf_extract_tables(content: bytes, extractor: str = "camelot", flavor: str = "lattice", min_conf: float = 0.5) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Try to extract tables; return (extracted_any, normalized_table_blocks).
    Each block: {type:"table", id, page, data, bbox?}
    """
    tables_out: List[Dict[str, Any]] = []
    found_any = False

    # Camelot path
    if extractor == "camelot" and _HAS_CAMELOT:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(content)
            tmp.flush()
            try:
                # flavor: lattice|stream
                tables = camelot.read_pdf(tmp.name, pages="all", flavor=flavor)
                order = 0
                for t in tables:
                    # Camelot parsing_report has accuracy; default to 1.0 if missing
                    acc = 1.0
                    try:
                        acc = float(t.parsing_report.get("accuracy") or 1.0)
                    except Exception:
                        pass
                    if acc < min_conf * 100.0:  # camelot uses percentage (0..100)
                        continue
                    data = [list(map(str, row)) for row in t.df.values.tolist()]
                    order += 1
                    bid = f"p{t.page}_t{order}_{hashlib.sha1(json.dumps(data).encode('utf-8')).hexdigest()[:4]}"
                    block: Dict[str, Any] = {"type": "table", "id": bid, "page": t.page, "data": data}
                    try:
                        if hasattr(t, "_bbox") and isinstance(t._bbox, (list, tuple)) and len(t._bbox) == 4:
                            block["bbox"] = list(map(float, t._bbox))
                    except Exception:
                        pass
                    tables_out.append(block)
                found_any = len(tables_out) > 0
                return found_any, tables_out
            except Exception:
                pass  # fall through to tabula

    # Tabula path
    if extractor in ("tabula", "camelot") and _HAS_TABULA:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(content)
            tmp.flush()
            try:
                dfs = tabula.read_pdf(tmp.name, pages="all", multiple_tables=True)
                order = 0
                for df in dfs or []:
                    data = [list(map(lambda x: "" if x is None else str(x), row)) for row in df.values.tolist()]
                    if not any(any(cell for cell in row) for row in data):
                        continue
                    order += 1
                    bid = f"p?_t{order}_{hashlib.sha1(json.dumps(data).encode('utf-8')).hexdigest()[:4]}"
                    tables_out.append({"type": "table", "id": bid, "page": None, "data": data})
                found_any = len(tables_out) > 0
                return found_any, tables_out
            except Exception:
                pass

    return False, []


# ---------------- JSON-RPC Handler ----------------

async def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Methods:
      - crawl.fetch {
          url, mode: "html"|"pdf", timeout_ms?, if_modified_since?, etag?, respect_robots?,
          pdf_table_extractor? ("camelot"|"tabula"|"none"),
          pdf_flavor? ("lattice"|"stream"),
          pdf_min_confidence? (0..1)
        }
        -> HTML: { content: string, metadata:{...}, jsonld:[...] }
           PDF : { content: string|content_base64, metadata:{...},
                   pdf_tables_extracted: bool, tables?: [...] }

      - crawl.canon { html, base_url, cleaner_engine?("bs4"|"trafilatura"|"justext"), keep_toc? }
        -> { blocks:[...], metadata:{...} }
    """
    mid = req.get("id")
    method = req.get("method")
    p = req.get("params") or {}

    def ok(result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code: str, message: str, hint: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        e: Dict[str, Any] = {"code": code, "message": message}
        if hint:
            e["hint"] = hint
        out = {"jsonrpc": "2.0", "id": mid, "error": e}
        if meta:
            out["meta"] = meta
        return out

    if method == "crawl.fetch":
        url = p.get("url")
        if not isinstance(url, str) or not url:
            return err("invalid_params", "url is required and must be a string")
        mode = (p.get("mode") or "html").lower()
        timeout = int(p.get("timeout_ms", 15000))
        ims = p.get("if_modified_since")
        et = p.get("etag")
        respect_robots = bool(p.get("respect_robots", True))

        content, meta, e = await _fetch(url, timeout_ms=timeout, ims=ims, etag=et, respect_robots=respect_robots)
        if e:
            return {"jsonrpc": "2.0", "id": mid, "error": e, "meta": meta}

        if content is None:
            return err("fetch_empty", "No content returned", meta=meta)

        # Auto-detect PDF by content-type if caller didn't specify
        if mode not in ("html", "pdf"):
            mode = "pdf" if (meta.get("mime") == "application/pdf") else "html"

        if mode == "pdf":
            extractor = (p.get("pdf_table_extractor") or "camelot").lower()
            flavor = (p.get("pdf_flavor") or "lattice").lower()
            min_conf = float(p.get("pdf_min_confidence", 0.5))
            extracted, tables = (False, [])
            if extractor != "none":
                extracted, tables = _pdf_extract_tables(content, extractor=extractor, flavor=flavor, min_conf=min_conf)

            text = _pdf_to_text(content) if not extracted else ""
            meta["mime"] = "application/pdf"
            res: Dict[str, Any] = {"metadata": meta, "pdf_tables_extracted": bool(extracted)}
            if extracted and tables:
                res["tables"] = tables
                # Prefer returning text too if available (non-fatal if empty)
                if text:
                    res["content"] = text
                else:
                    # If no text, provide base64 so caller can store original
                    res["content_base64"] = base64.b64encode(content).decode("ascii")
            else:
                if text:
                    res["content"] = text
                else:
                    res["content_base64"] = base64.b64encode(content).decode("ascii")
                    res["note"] = "returned base64 because text extraction failed"
            return ok(res)

        # HTML mode
        html = content.decode("utf-8", "ignore")
        result: Dict[str, Any] = {"content": html, "metadata": meta}
        try:
            result["jsonld"] = _extract_jsonld(html, meta.get("final_url") or url)
        except Exception:
            pass
        return ok(result)

    if method == "crawl.canon":
        html = p.get("html")
        base_url = p.get("base_url")
        if not isinstance(html, str) or not html:
            return err("invalid_params", "html must be a non-empty string")
        if not isinstance(base_url, str) or not base_url:
            return err("invalid_params", "base_url must be a non-empty string")
        cleaner_engine = (p.get("cleaner_engine") or "bs4").lower()
        keep_toc = bool(p.get("keep_toc", True))
        blocks, extra = _canon_html(html, base_url, cleaner_engine=cleaner_engine, keep_toc=keep_toc)
        return ok({"blocks": blocks, "metadata": extra})

    return err("method_not_found", str(method))
