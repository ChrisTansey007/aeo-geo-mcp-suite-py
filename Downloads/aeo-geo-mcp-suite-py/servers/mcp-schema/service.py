# schema_service.py
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jsonschema import Draft202012Validator, exceptions as js_exceptions

# ---------- Template env (env-overridable) ----------

_DEFAULT_TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"
TEMPLATES_DIR = pathlib.Path(os.getenv("SCHEMA_TEMPLATES_DIR", str(_DEFAULT_TEMPLATES_DIR)))

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)

# Jinja helpers
env.filters["json"] = lambda v: json.dumps(v, ensure_ascii=False)
env.filters["inc"] = lambda i: int(i) + 1  # for Breadcrumb/ItemList positions


def _tpl_name(typ: str, variant: Optional[str]) -> str:
    """Resolve template name with variant fallback."""
    if variant:
        cand = f"{typ}.{variant}.jinja"
        if (TEMPLATES_DIR / cand).exists():
            return cand
    base = f"{typ}.jinja"
    if (TEMPLATES_DIR / base).exists():
        return base
    raise FileNotFoundError(f"Template not found for type='{typ}', variant='{variant or ''}'")


def _sort_keys_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_keys_recursive(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sort_keys_recursive(v) for v in obj]
    return obj


def _normalize_jsonld(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj.setdefault("@context", "https://schema.org")
    t = obj.get("@type")
    if isinstance(t, list) and len(t) == 1:
        obj["@type"] = t[0]
    return obj


def _render_template(typ: str, data: Dict[str, Any], variant: Optional[str]) -> Dict[str, Any]:
    name = _tpl_name(typ, variant)
    t = env.get_template(name)
    text = t.render(**data)
    out = json.loads(text)
    out = _normalize_jsonld(out)
    return _sort_keys_recursive(out)


# ---------- Minimal JSON Schemas per type (Draft 2020-12) ----------

SCHEMA_MAP: Dict[str, Dict[str, Any]] = {
    "LocalBusiness": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["@context", "@type", "name"],
        "properties": {
            "@context": {"const": "https://schema.org"},
            "@type": {
                "anyOf": [
                    {"const": "LocalBusiness"},
                    {"type": "array", "contains": {"const": "LocalBusiness"}},
                ]
            },
            "name": {"type": "string", "minLength": 1},
            "url": {"type": "string"},
            "telephone": {"type": "string"},
            "image": {"anyOf": [{"type": "string"}, {"type": "object"}]},
            "address": {
                "type": "object",
                "required": [
                    "@type",
                    "streetAddress",
                    "addressLocality",
                    "addressRegion",
                    "postalCode",
                    "addressCountry",
                ],
                "properties": {
                    "@type": {"const": "PostalAddress"},
                    "streetAddress": {"type": "string"},
                    "addressLocality": {"type": "string"},
                    "addressRegion": {"type": "string"},
                    "postalCode": {"type": "string"},
                    "addressCountry": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "geo": {
                "type": "object",
                "required": ["@type", "latitude", "longitude"],
                "properties": {
                    "@type": {"const": "GeoCoordinates"},
                    "latitude": {"type": ["number", "string"]},
                    "longitude": {"type": ["number", "string"]},
                },
                "additionalProperties": True,
            },
            "sameAs": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    },
    "Product": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["@context", "@type", "name"],
        "properties": {
            "@context": {"const": "https://schema.org"},
            "@type": {"const": "Product"},
            "name": {"type": "string", "minLength": 1},
            "sku": {"type": "string"},
            "brand": {"anyOf": [{"type": "string"}, {"type": "object"}]},
            "offers": {
                "type": "object",
                "required": ["@type", "price", "priceCurrency"],
                "properties": {
                    "@type": {"const": "Offer"},
                    "price": {"type": ["string", "number"]},
                    "priceCurrency": {"type": "string"},
                    "availability": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    },
    "Article": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["@context", "@type", "headline", "datePublished"],
        "properties": {
            "@context": {"const": "https://schema.org"},
            "@type": {"const": "Article"},
            "headline": {"type": "string", "minLength": 1},
            "author": {"anyOf": [{"type": "string"}, {"type": "object"}]},
            "datePublished": {"type": "string"},
            "image": {"anyOf": [{"type": "string"}, {"type": "array"}]},
            "mainEntityOfPage": {"type": ["string", "object"]},
        },
        "additionalProperties": True,
    },
    "FAQPage": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["@context", "@type", "mainEntity"],
        "properties": {
            "@context": {"const": "https://schema.org"},
            "@type": {"const": "FAQPage"},
            "mainEntity": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["@type", "name", "acceptedAnswer"],
                    "properties": {
                        "@type": {"const": "Question"},
                        "name": {"type": "string"},
                        "acceptedAnswer": {
                            "type": "object",
                            "required": ["@type", "text"],
                            "properties": {
                                "@type": {"const": "Answer"},
                                "text": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "additionalProperties": True,
    },
    "BreadcrumbList": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["@context", "@type", "itemListElement"],
        "properties": {
            "@context": {"const": "https://schema.org"},
            "@type": {"const": "BreadcrumbList"},
            "itemListElement": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["@type", "position", "item"],
                    "properties": {
                        "@type": {"const": "ListItem"},
                        "position": {"type": "integer", "minimum": 1},
                        "item": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                    },
                },
            },
        },
        "additionalProperties": True,
    },
    "ItemList": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["@context", "@type", "itemListElement"],
        "properties": {
            "@context": {"const": "https://schema.org"},
            "@type": {"const": "ItemList"},
            "itemListElement": {"type": "array", "minItems": 1},
        },
        "additionalProperties": True,
    },
}


def _detect_primary_type(jsonld: Dict[str, Any]) -> Optional[str]:
    t = jsonld.get("@type")
    if isinstance(t, str):
        return t
    if isinstance(t, list) and t:
        if "LocalBusiness" in t:
            return "LocalBusiness"
        return str(t[0])
    return None


def _validator_for(jsonld: Dict[str, Any]) -> Optional[Draft202012Validator]:
    ptype = _detect_primary_type(jsonld)
    schema = SCHEMA_MAP.get(ptype or "")
    if not schema:
        return None
    return Draft202012Validator(schema)


# ---------- Suggestions (best practices) ----------

def _suggestions(jsonld: Dict[str, Any]) -> List[str]:
    t = _detect_primary_type(jsonld)
    s: List[str] = []
    if not t:
        s.append("Set @type (e.g., 'LocalBusiness', 'Product').")
        return s

    if t == "LocalBusiness":
        if "telephone" not in jsonld:
            s.append("Add 'telephone' (E.164 format preferred).")
        addr = jsonld.get("address")
        if not isinstance(addr, dict) or addr.get("@type") != "PostalAddress":
            s.append("Include 'address' as a PostalAddress with street/city/region/postalCode/country.")
        if "geo" not in jsonld and "hasMap" not in jsonld:
            s.append("Add 'geo' (GeoCoordinates) or 'hasMap' URL.")
        if "sameAs" not in jsonld:
            s.append("Add 'sameAs' links to official profiles (e.g., GMB, Facebook).")

    if t == "Product":
        offers = jsonld.get("offers")
        if offers and "priceCurrency" not in offers:
            s.append("Include 'priceCurrency' with 'price' in offers.")
        if "brand" not in jsonld:
            s.append("Add 'brand' as a Brand object or string.")

    if t == "Article":
        if "author" not in jsonld and "publisher" not in jsonld:
            s.append("Add 'author' or 'publisher'.")
        if "image" not in jsonld:
            s.append("Add 'image' for better eligibility in rich results.")

    if t == "FAQPage":
        main = jsonld.get("mainEntity") or []
        if isinstance(main, list) and len(main) < 1:
            s.append("Add at least one Q&A in 'mainEntity'.")

    if t == "BreadcrumbList":
        items = jsonld.get("itemListElement") or []
        positions = [i.get("position") for i in items if isinstance(i, dict)]
        if positions and positions != list(range(1, len(positions) + 1)):
            s.append("Ensure breadcrumb 'position' starts at 1 and is contiguous.")

    if t == "ItemList":
        items = jsonld.get("itemListElement") or []
        if isinstance(items, list) and len(items) > 5 and "itemListOrder" not in jsonld:
            s.append("Consider adding 'itemListOrder' (e.g., 'http://schema.org/ItemListOrderAscending').")

    return s


# ---------- Google-aware warnings ----------

def _google_warnings(jsonld: Dict[str, Any], page_url: Optional[str], page_path: Optional[str]) -> List[str]:
    warns: List[str] = []
    typ = _detect_primary_type(jsonld)

    # HowTo: warn due to limited eligibility (kept generic; exact policy evolves)
    if typ == "HowTo":
        warns.append("HowTo rich results are limited; consider alternative patterns unless strictly necessary.")

    # FAQPage: warn if not on a '/faq' route (heuristic)
    if typ == "FAQPage":
        path = page_path
        if not path and page_url:
            try:
                path = urlparse(page_url).path or ""
            except Exception:
                path = ""
        if path and "/faq" not in path.lower():
            warns.append("FAQPage is usually eligible only on dedicated FAQ pages; consider routing under '/faq'.")

    return warns


# ---------- Registry enrichment ----------

def _load_registry() -> Dict[str, Any]:
    """
    Flexible loader for a registry YAML. Expected (any of):
      - { areaServed: [ "City A", "City B", ... ], services: [ "Fence Repair", ... ] }
      - { markets: { <market_key>: { city:[...], neighbors:[...], county:[...] } }, services:[...] }
    Env: SCHEMA_REGISTRY_PATH, LOCAL_MARKET (preferred), GEO_MARKET (fallback, for markets.* selection)
    """
    path = os.getenv("SCHEMA_REGISTRY_PATH")
    if not path:
        # Try a few sane defaults
        candidates = [
            pathlib.Path(__file__).parents[2] / "packages" / "geo" / "registry.yaml",
            pathlib.Path(__file__).parents[2] / "packages" / "geo" / "aliases.yaml",
        ]
        for c in candidates:
            if c.exists():
                path = str(c)
                break
    if not path:
        return {}
    try:
        return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _enrich_organization(jsonld: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(jsonld)  # shallow copy

    # areaServed
    if "areaServed" not in out:
        served: List[str] = []
        if isinstance(registry.get("areaServed"), list):
            served = [str(x) for x in registry["areaServed"] if x]
        elif isinstance(registry.get("markets"), dict):
            mk = os.getenv("LOCAL_MARKET") or os.getenv("GEO_MARKET") or "wilmington_nc"
            m = registry["markets"].get(mk) or {}
            for key in ("city", "neighbors", "county"):
                vals = m.get(key) or []
                served += [str(x) for x in vals if x]
        if served:
            out["areaServed"] = [{"@type": "Place", "name": n} for n in served]

    # hasOfferCatalog (simplified)
    if "hasOfferCatalog" not in out:
        services: List[str] = []
        if isinstance(registry.get("services"), list):
            services = [str(x) for x in registry["services"] if x]
        if services:
            out["hasOfferCatalog"] = {
                "@type": "OfferCatalog",
                "name": "Services",
                "itemListElement": [
                    {"@type": "Offer", "itemOffered": {"@type": "Service", "name": s}} for s in services
                ],
            }

    return _sort_keys_recursive(out)


# ---------- Public API helpers ----------

def generate(params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Returns (jsonld, warnings, suggestions).
    Supports: { type, data?, variant?, enrich?: { registry?: bool } }
    """
    typ = params.get("type")
    if not typ or not isinstance(typ, str):
        raise ValueError("type is required and must be a string")
    data = params.get("data") or {}
    if not isinstance(data, dict):
        raise ValueError("data must be an object")
    variant = params.get("variant")
    enrich = params.get("enrich") or {}

    out = _render_template(typ, data, variant)

    # Optional enrichment for Organization
    warnings: List[str] = []
    if typ == "Organization" and enrich.get("registry"):
        reg = _load_registry()
        out = _enrich_organization(out, reg)

    suggestions = _suggestions(out)
    return out, warnings, suggestions


def validate_jsonld(
    jsonld: Dict[str, Any],
    checks: Optional[List[str]] = None,
    page_url: Optional[str] = None,
    page_path: Optional[str] = None,
) -> Tuple[bool, List[Dict[str, str]], List[str], List[str]]:
    """
    Return (valid, errors[], warnings[], suggestions[]).
    checks: subset of {"schema_org","google"}
    """
    checks = checks or ["schema_org"]

    if not isinstance(jsonld, dict):
        return False, [{"path": "/", "message": "jsonld must be an object", "rule": "core.type"}], [], []

    # Core minimum
    core_errors: List[Dict[str, str]] = []
    if "@context" not in jsonld:
        core_errors.append({"path": "/", "message": "@context required", "rule": "core.required"})
    if "@type" not in jsonld:
        core_errors.append({"path": "/", "message": "@type required", "rule": "core.required"})
    if core_errors:
        return False, core_errors, [], _suggestions(jsonld)

    errors: List[Dict[str, str]] = []
    warnings: List[str] = []

    # 1) Schema.org (our pragmatic contracts)
    if "schema_org" in checks:
        v = _validator_for(jsonld)
        if v:
            for err in sorted(v.iter_errors(jsonld), key=lambda e: e.path):
                path = "/" + "/".join(map(str, err.path)) if err.path else "/"
                errors.append({"path": path, "message": err.message, "rule": "schema.validation"})

    # 2) Google-aware lints
    if "google" in checks:
        warnings += _google_warnings(jsonld, page_url, page_path)

    return (len(errors) == 0), errors, warnings, _suggestions(jsonld)


# ---------- JSON-RPC helpers ----------

def _ok(id_: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}

def _err(id_: Any, code: str, message: str, hint: Optional[str] = None) -> Dict[str, Any]:
    e: Dict[str, Any] = {"code": code, "message": message}
    if hint:
        e["hint"] = hint
    return {"jsonrpc": "2.0", "id": id_, "error": e}


# ---------- JSON-RPC handler ----------

async def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Methods:
      schema.generate { type, data?, variant?, enrich? }
        -> { jsonld, warnings[], suggestions[] }

      schema.validate { jsonld, checks?:["schema_org","google"], page_url?, page_path? }
        -> { valid, errors:[{path,message,rule}], warnings:[], suggestions:[], rich_results_test_url? }
    """
    mid = req.get("id")
    method = req.get("method")
    p = req.get("params", {}) or {}

    try:
        if method == "schema.generate":
            jsonld, warnings, suggestions = generate(p)
            return _ok(mid, {"jsonld": jsonld, "warnings": warnings, "suggestions": suggestions})

        if method == "schema.validate":
            jsonld = p.get("jsonld")
            checks = p.get("checks")
            page_url = p.get("page_url")
            page_path = p.get("page_path")
            valid, errors, warnings, suggestions = validate_jsonld(jsonld, checks=checks, page_url=page_url, page_path=page_path)
            out: Dict[str, Any] = {
                "valid": valid,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
            }
            if page_url:
                out["rich_results_test_url"] = f"https://search.google.com/test/rich-results?url={quote_plus(page_url)}"
            return _ok(mid, out)

        return _err(mid, "method_not_found", str(method))

    except FileNotFoundError as e:
        return _err(mid, "template_not_found", str(e))
    except (ValueError, TypeError) as e:
        return _err(mid, "invalid_params", str(e))
    except js_exceptions.SchemaError as e:
        return _err(mid, "schema_error", f"Validator schema error: {getattr(e, 'message', str(e))}")
    except Exception as e:
        return _err(mid, "internal", str(e))
