# publisher_service.py
from __future__ import annotations

import asyncio
import datetime as dt
import difflib
import hashlib
import json
import os
import pathlib
import re
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import httpx
from git import Actor, Repo
from slugify import slugify
from urllib.parse import quote_plus

# ----------------- Helpers -----------------

def _slug_city(city: str) -> str:
    return slugify(city)

def _slug_service_city(service: str, city: str) -> str:
    return f"{slugify(service)}/{slugify(city)}"

def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def _etag_for(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
    return h.hexdigest()

def _safe_target(base_repo: str, rel_path: str) -> pathlib.Path:
    base = pathlib.Path(base_repo).resolve()
    target = (base / rel_path.lstrip("/")).resolve()
    if not str(target).startswith(str(base) + os.sep) and target != base:
        raise ValueError("path_traversal")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

def _read_text_if_exists(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

def _atomic_write_text(p: pathlib.Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(p.parent), encoding="utf-8") as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, str(p))

def _unified_diff(old: str, new: str, rel: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a{rel}",
            tofile=f"b{rel}",
        )
    )

def _map_url(base_url: str, rel: str) -> str:
    base_url = base_url.rstrip("/")
    rel = "/" + rel.lstrip("/")
    # Map .md to directory URL; keep .html as-is; others raw
    if rel.endswith("/index.md"):
        return f"{base_url}{rel[:-9]}/"
    if rel.endswith(".md"):
        return f"{base_url}{rel[:-3]}/"
    return f"{base_url}{rel}"

def _maybe_embed_jsonld_into_html(html: str, jsonld: Any) -> str:
    if not jsonld:
        return html
    payload = _json_dumps_stable(jsonld)
    script = f'<script type="application/ld+json">{payload}</script>'
    # Prefer in <head> if present, else before </body>, else append
    if "</head>" in html:
        return html.replace("</head>", script + "\n</head>")
    if "</body>" in html:
        return html.replace("</body>", script + "\n</body>")
    return html + ("\n" if not html.endswith("\n") else "") + script + "\n"

def _ensure_repo(path: str) -> Repo:
    repo_path = pathlib.Path(path)
    if not (repo_path / ".git").exists():
        repo = Repo.init(path)
    else:
        repo = Repo(path)
    # basic config (safe defaults)
    user_name = os.getenv("GIT_USER_NAME", "Publisher Bot")
    user_email = os.getenv("GIT_USER_EMAIL", "publisher@example.com")
    repo.config_writer().set_value("user", "name", user_name).release()
    repo.config_writer().set_value("user", "email", user_email).release()
    # signing if provided
    signing_key = os.getenv("GIT_SIGNING_KEY")
    if signing_key:
        cw = repo.config_writer()
        cw.set_value("user", "signingkey", signing_key)
        cw.set_value("commit", "gpgsign", "true")
        cw.release()
    return repo

def _semantic_message(rel: str, draft: bool, changed_files: List[str], user_msg: Optional[str]) -> str:
    if user_msg:
        return user_msg
    scope = "publish"
    detail = f"upsert {rel} [draft={str(draft).lower()}]"
    if len(changed_files) > 1:
        detail += f" (+{len(changed_files)-1} file{'s' if len(changed_files)>2 else ''})"
    return f"feat({scope}): {detail}"

def _short(v: str) -> str:
    return v[:7]

def _content_version(rel: str, main_content: str, jsonld: Any) -> str:
    jl = _json_dumps_stable(jsonld) if jsonld is not None else ""
    return _short(_etag_for(rel, main_content, jl))

def _now_iso_date() -> str:
    # sitemap best-practice: date or dateTime; we choose full UTC dateTime
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# --------------- Sitemap utilities ---------------

def _update_sitemap_lastmod(sitemap_path: pathlib.Path, page_url: str) -> bool:
    """
    Update/insert <lastmod> for the given page_url within sitemap.xml.
    Returns True if file was modified.
    """
    if not sitemap_path.exists():
        return False
    try:
        ET.register_namespace("", "http://www.sitemaps.org/schemas/sitemap/0.9")
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        tree = ET.parse(str(sitemap_path))
        root = tree.getroot()
        modified = False
        # find <url><loc> == page_url
        for url in root.findall("sm:url", ns):
            loc = url.find("sm:loc", ns)
            if loc is not None and (loc.text or "").strip() == page_url:
                lastmod = url.find("sm:lastmod", ns)
                if lastmod is None:
                    lastmod = ET.SubElement(url, "{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod")
                new_val = _now_iso_date()
                if (lastmod.text or "") != new_val:
                    lastmod.text = new_val
                    modified = True
                break
        if modified:
            tmp = sitemap_path.with_suffix(sitemap_path.suffix + ".tmp")
            tree.write(str(tmp), encoding="utf-8", xml_declaration=True)
            os.replace(str(tmp), str(sitemap_path))
        return modified
    except Exception:
        return False

# --------------- Pings (IndexNow + optional legacy) ---------------

async def _bounded_fetch(client: httpx.AsyncClient, method: str, url: str, **kw) -> httpx.Response:
    retries = int(os.getenv("PING_RETRIES", "2"))
    for attempt in range(retries + 1):
        try:
            r = await client.request(method, url, timeout=10.0, **kw)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            return r
        except Exception:
            if attempt >= retries:
                raise

async def _ping_endpoints(urls: List[str], sitemap_url: Optional[str], indexnow: bool, google_ping_enabled: bool) -> Dict[str, Any]:
    """
    urls: content URLs to notify (IndexNow)
    sitemap_url: sitemap URL for legacy Google/Bing ping if enabled
    """
    results = {"submitted_to": [], "indexnow_status": None, "google_ping": None, "bing_ping": None}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # IndexNow
        if indexnow and urls:
            key = os.getenv("INDEXNOW_KEY")
            key_loc = os.getenv("INDEXNOW_KEY_LOCATION", "")
            if key:
                # host must be site host (first URL host)
                try:
                    host = re.sub(r"^https?://", "", urls[0], flags=re.I).split("/")[0]
                except Exception:
                    host = ""
                payload = {"host": host, "key": key, "keyLocation": key_loc, "urlList": urls}
                try:
                    r = await _bounded_fetch(client, "POST", "https://api.indexnow.org/indexnow", json=payload)
                    results["submitted_to"].append("indexnow")
                    results["indexnow_status"] = r.status_code
                except Exception as e:
                    results["indexnow_status"] = f"error:{e}"
        # Legacy ping with sitemap (optional)
        if google_ping_enabled and sitemap_url:
            try:
                g = f"https://www.google.com/ping?sitemap={quote_plus(sitemap_url)}"
                b = f"https://www.bing.com/ping?sitemap={quote_plus(sitemap_url)}"
                rg = await _bounded_fetch(client, "GET", g)
                rb = await _bounded_fetch(client, "GET", b)
                results["submitted_to"] += ["google_ping", "bing_ping"]
                results["google_ping"] = rg.status_code
                results["bing_ping"] = rb.status_code
            except Exception as e:
                results["google_ping"] = f"error:{e}"
                results["bing_ping"] = f"error:{e}"
    return results

# --------------- JSON-RPC helpers ---------------

def _ok(id_: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}

def _err(id_: Any, code: str, message: str, hint: Optional[str] = None) -> Dict[str, Any]:
    e: Dict[str, Any] = {"code": code, "message": message}
    if hint:
        e["hint"] = hint
    return {"jsonrpc": "2.0", "id": id_, "error": e}

# --------------- JSON-RPC Handler ---------------

async def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Methods:
      publish.upsert {
        path, markdown?|html?, jsonld?, draft?, dry_run?, message?, embed_jsonld?, preview?
      } -> {
        status: "ok"|"not_modified",
        diff, url, version, etag, committed?: bool,
        sitemap_updated?: bool
      }

      publish.rollback { path, version, dry_run? } -> { status, url, diff? }

      publish.ping { url?, urls?, indexnow?, sitemap_url? } ->
        { ok: bool, submitted_to:[], details:{...} }

      publish.precheck { url?|html, jsonld? } ->
        { schema_ok: bool, google_warnings:[], render_ok: bool, rich_results_test_url?: string }
    """
    mid = req.get("id")
    method = req.get("method")
    p = req.get("params", {}) or {}

    repo_path = os.getenv("REPO_PATH", p.get("repo", "./content-repo"))
    base_url = os.getenv("BASE_URL", p.get("baseUrl", "https://example.com"))
    preview_dir = os.getenv("PREVIEW_DIR", "preview")  # used when preview=true
    sitemap_path_env = os.getenv("SITEMAP_PATH")  # optional absolute or repo-relative path

    if method == "publish.upsert":
        rel: str = p.get("path") or ""
        if not isinstance(rel, str) or not rel.strip():
            return _err(mid, "invalid_params", "path is required")

        markdown = p.get("markdown")
        html = p.get("html")
        jsonld = p.get("jsonld")
        draft = bool(p.get("draft", False))
        dry = bool(p.get("dry_run", False))
        message = p.get("message")
        embed_jsonld = bool(p.get("embed_jsonld", False))
        preview = bool(p.get("preview", False))

        if (markdown is None) and (html is None):
            return _err(mid, "invalid_params", "Provide 'markdown' or 'html'")
        if (markdown is not None) and (html is not None):
            return _err(mid, "invalid_params", "Provide only one of 'markdown' or 'html'")

        # When preview, write under preview_dir
        rel_effective = rel
        if preview:
            rel_effective = f"{preview_dir.rstrip('/')}/{rel.lstrip('/')}"

        try:
            target = _safe_target(repo_path, rel_effective)
        except ValueError:
            return _err(mid, "invalid_path", "Path escapes repository root")

        before = _read_text_if_exists(target)

        if markdown is not None:
            after = markdown
        else:
            after = html or ""
            if embed_jsonld and jsonld:
                after = _maybe_embed_jsonld_into_html(after, jsonld)

        # deterministic version across main content + jsonld
        version = _content_version(rel_effective, after, jsonld)
        etag = version

        diff = _unified_diff(before, after, rel_effective)
        changed = before != after
        url = _map_url(base_url, rel_effective)

        if dry:
            return _ok(mid, {
                "status": "ok" if changed else "not_modified",
                "diff": diff, "url": url, "version": version, "etag": etag, "committed": False,
                "sitemap_updated": False
            })

        if not changed:
            return _ok(mid, {
                "status": "not_modified", "diff": "", "url": url, "version": version, "etag": etag, "committed": False,
                "sitemap_updated": False
            })

        # Write content
        try:
            _atomic_write_text(target, after)
        except Exception as e:
            return _err(mid, "write_failed", f"Failed to write content: {e}")

        # JSON-LD sidecar (if provided and not embedded)
        changed_files: List[str] = [str(target.relative_to(repo_path))]
        if jsonld and not embed_jsonld:
            sidecar_path = target.with_suffix(target.suffix + ".jsonld")
            try:
                _atomic_write_text(sidecar_path, json.dumps(jsonld, ensure_ascii=False, indent=2))
                changed_files.append(str(sidecar_path.relative_to(repo_path)))
            except Exception as e:
                return _err(mid, "write_failed", f"Failed to write JSON-LD: {e}")

        # Git commit
        try:
            repo = _ensure_repo(repo_path)
            repo.index.add(changed_files)
            actor = Actor(os.getenv("GIT_USER_NAME", "Publisher Bot"), os.getenv("GIT_USER_EMAIL", "publisher@example.com"))
            commit_msg = _semantic_message(rel_effective, draft or preview, changed_files, message)
            repo.index.commit(commit_msg, author=actor, committer=actor)
        except Exception as e:
            return _err(mid, "git_error", f"Git commit failed: {e}")

        # Optional: update sitemap lastmod
        sitemap_updated = False
        try:
            if sitemap_path_env:
                sp = pathlib.Path(sitemap_path_env)
                if not sp.is_absolute():
                    sp = pathlib.Path(repo_path) / sp
                sitemap_updated = _update_sitemap_lastmod(sp, url)
        except Exception:
            sitemap_updated = False

        return _ok(mid, {
            "status": "ok",
            "diff": diff,
            "url": url,
            "version": version,
            "etag": etag,
            "committed": True,
            "sitemap_updated": bool(sitemap_updated)
        })

    if method == "publish.rollback":
        rel: str = p.get("path") or ""
        version: str = p.get("version") or ""
        dry = bool(p.get("dry_run", False))
        if not rel or not version:
            return _err(mid, "invalid_params", "path and version are required")

        try:
            target = _safe_target(repo_path, rel)
        except ValueError:
            return _err(mid, "invalid_path", "Path escapes repository root")

        try:
            repo = _ensure_repo(repo_path)
        except Exception as e:
            return _err(mid, "git_error", f"Open repo failed: {e}")

        try:
            commits = list(repo.iter_commits(paths=rel, max_count=200))
        except Exception as e:
            return _err(mid, "git_error", f"Iter commits failed: {e}")

        match_content: Optional[str] = None
        for c in commits:
            try:
                blob = (c.tree / rel).data_stream.read()
                content = blob.decode("utf-8", "ignore")
                if _content_version(rel, content, None) == version:
                    match_content = content
                    break
            except Exception:
                continue

        if match_content is None:
            return _err(mid, "not_found", "Requested version not found for this path")

        before = _read_text_if_exists(target)
        diff = _unified_diff(before, match_content, rel)
        url = _map_url(base_url, rel)

        if dry:
            return _ok(mid, {"status": "preview", "url": url, "diff": diff})

        try:
            _atomic_write_text(target, match_content)
            repo.index.add([str(pathlib.Path(rel))])
            actor = Actor(os.getenv("GIT_USER_NAME", "Publisher Bot"), os.getenv("GIT_USER_EMAIL", "publisher@example.com"))
            repo.index.commit(f"revert: {rel} -> {version}", author=actor, committer=actor)
        except Exception as e:
            return _err(mid, "git_error", f"Rollback commit failed: {e}")

        return _ok(mid, {"status": "ok", "url": url})

    if method == "publish.ping":
        # { url?, urls?, indexnow?, sitemap_url? }
        url_single = p.get("url")
        urls_list = p.get("urls") or ([] if not url_single else [url_single])
        indexnow = bool(p.get("indexnow", False))
        sitemap_url = p.get("sitemap_url")  # if you want to ping search engines with a sitemap
        google_ping_enabled = bool(os.getenv("GOOGLE_PING_ENABLED", "false").lower() in ("1", "true", "yes"))

        try:
            details = await _ping_endpoints(urls_list, sitemap_url, indexnow, google_ping_enabled)
            ok = True
            # consider ok if any target accepted
            if not details.get("submitted_to"):
                ok = False
            return _ok(mid, {"ok": ok, "submitted_to": details.get("submitted_to", []), "details": details})
        except Exception as e:
            return _ok(mid, {"ok": False, "submitted_to": [], "details": {"error": str(e)}})

    if method == "publish.precheck":
        """
        params: { url? | html, jsonld? }
        """
        html = p.get("html")
        url = p.get("url")
        jsonld = p.get("jsonld")
        doc = None

        if not html and not url:
            return _err(mid, "invalid_params", "Provide 'html' or 'url'")

        # fetch if url provided
        if url and not html:
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                    r = await client.get(url, headers={"User-Agent": os.getenv("CRAWL_UA", "AEO-GEO-MCP-Crawler/1.2")})
                    r.raise_for_status()
                    html = r.text
            except Exception as e:
                return _ok(mid, {"schema_ok": False, "render_ok": False, "google_warnings": [f"fetch_failed:{e}"]})

        render_ok = bool(html and ("</" in html or "<" in html))

        # Minimal JSON-LD checks: parse embedded and/or provided jsonld
        google_warnings: List[str] = []
        schema_ok = True

        embedded_ok = True
        try:
            import bs4  # local import to avoid hard dep at module import time
            soup = bs4.BeautifulSoup(html or "", "lxml")
            for tag in soup.select('script[type="application/ld+json"]'):
                try:
                    data = json.loads(tag.text or tag.string or "")
                    if isinstance(data, list):
                        for item in data:
                            if "@context" not in item or "@type" not in item:
                                embedded_ok = False
                    else:
                        if "@context" not in data or "@type" not in data:
                            embedded_ok = False
                except Exception:
                    embedded_ok = False
        except Exception:
            # If BS4 not available, skip embedded check
            embedded_ok = True

        provided_ok = True
        if jsonld is not None:
            try:
                if isinstance(jsonld, str):
                    obj = json.loads(jsonld)
                else:
                    obj = jsonld
                # accept list or object
                if isinstance(obj, list):
                    for it in obj:
                        if not isinstance(it, dict) or "@context" not in it or "@type" not in it:
                            provided_ok = False
                elif isinstance(obj, dict):
                    if "@context" not in obj or "@type" not in obj:
                        provided_ok = False
                else:
                    provided_ok = False
            except Exception:
                provided_ok = False

        schema_ok = embedded_ok and provided_ok
        if not embedded_ok:
            google_warnings.append("Embedded JSON-LD missing @context/@type or not parseable.")
        if not provided_ok and jsonld is not None:
            google_warnings.append("Provided JSON-LD missing @context/@type or not parseable.")

        out: Dict[str, Any] = {
            "schema_ok": bool(schema_ok),
            "render_ok": bool(render_ok),
            "google_warnings": google_warnings,
        }
        if url:
            out["rich_results_test_url"] = f"https://search.google.com/test/rich-results?url={quote_plus(url)}"
        return _ok(mid, out)

    return _err(mid, "method_not_found", str(method))
