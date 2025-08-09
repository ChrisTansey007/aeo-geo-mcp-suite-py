# analytics_service.py
from __future__ import annotations

import os
import json
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

API_VERSION = "1.1.0"
SERVICE_NAME = "mcp-analytics"

# ---------- DB init / connection ----------

_CONN: Optional[aiosqlite.Connection] = None

def _db_path() -> str:
    return os.getenv("AN_DB", os.getenv("DB_PATH", "./analytics.db"))

async def _ensure_columns(conn: aiosqlite.Connection) -> None:
    """
    Add new columns if they don't exist yet. Backward-compatible, no data loss.
    """
    # presence: +position (INT), +definition_version (TEXT)
    cur = await conn.execute("PRAGMA table_info(presence)")
    cols = {row[1] for row in await cur.fetchall()}  # row[1] = name
    if "position" not in cols:
        await conn.execute("ALTER TABLE presence ADD COLUMN position INTEGER")
    if "definition_version" not in cols:
        await conn.execute("ALTER TABLE presence ADD COLUMN definition_version TEXT")
    # kpi_points remains unchanged for now

async def _ensure_db() -> aiosqlite.Connection:
    """
    Lazily open a single shared aiosqlite connection, set PRAGMAs,
    and ensure schema + indexes exist.
    """
    global _CONN
    if _CONN:
        return _CONN

    _CONN = await aiosqlite.connect(_db_path())
    _CONN.row_factory = aiosqlite.Row

    # Pragmas: good perf & safe enough for analytics
    await _CONN.execute("PRAGMA journal_mode=WAL;")
    await _CONN.execute("PRAGMA synchronous=NORMAL;")
    await _CONN.execute("PRAGMA foreign_keys=ON;")

    await _CONN.executescript(
        """
CREATE TABLE IF NOT EXISTS presence (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query TEXT NOT NULL,
  engine TEXT NOT NULL,
  geo TEXT,
  serp_surface TEXT,
  engine_flavor TEXT,
  seen INTEGER NOT NULL CHECK (seen IN (0,1)),
  citations INTEGER,
  details TEXT,
  ts DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  UNIQUE(query, engine, geo, ts)
);

CREATE TABLE IF NOT EXISTS kpi_points (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  metric TEXT NOT NULL,
  t DATETIME NOT NULL,
  v REAL NOT NULL,
  geo TEXT,
  src TEXT NOT NULL DEFAULT 'csv',
  UNIQUE(metric, t, geo, src)
);

-- Helpful indexes for the common filters / sorts
CREATE INDEX IF NOT EXISTS idx_presence_engine_geo_ts
  ON presence (engine, geo, ts);
CREATE INDEX IF NOT EXISTS idx_presence_query_ts
  ON presence (query, ts);

CREATE INDEX IF NOT EXISTS idx_kpi_metric_geo_t
  ON kpi_points (metric, geo, t);
"""
    )
    await _ensure_columns(_CONN)
    await _CONN.commit()
    return _CONN

# ---------- Utilities ----------

def _iso_utc(dt_obj: dt.datetime) -> str:
    return dt_obj.replace(microsecond=0, tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")

def _parse_iso_utc(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        # Accept 'Z' or '+00:00'
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None

def _floor_time(d: dt.datetime, granularity: str) -> dt.datetime:
    g = (granularity or "second").lower()
    d = d.astimezone(dt.timezone.utc)
    if g == "day":
        return d.replace(hour=0, minute=0, second=0, microsecond=0)
    if g == "minute":
        return d.replace(second=0, microsecond=0)
    return d.replace(microsecond=0)  # 'second'

def _ok(id_: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}

def _err(id_: Any, code: str, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    e: Dict[str, Any] = {"code": code, "message": message}
    if data:
        e["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": e}

def _require(p: Dict[str, Any], keys: List[str]) -> Optional[Tuple[str, str]]:
    for k in keys:
        if p.get(k) is None:
            return ("invalid_params", f"Missing required parameter: {k}")
    return None

# ---------- Normalization helpers ----------

_SURFACE_MAP = {
    "ai overview": "ai_overview",
    "aio": "ai_overview",
    "ai_overview": "ai_overview",
    "copilot": "copilot_answer",
    "copilot answer": "copilot_answer",
    "copilot_answer": "copilot_answer",
    "perplexity": "perplexity_card",
    "perplexity card": "perplexity_card",
    "perplexity_card": "perplexity_card",
    "web": "web_answer",
    "web answer": "web_answer",
    "web_answer": "web_answer",
}

def _norm_engine(s: Optional[str]) -> Optional[str]:
    return s.lower() if isinstance(s, str) else s

def _norm_surface(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return s
    key = s.strip().lower()
    return _SURFACE_MAP.get(key, s.lower())

def _norm_flavor(s: Optional[str]) -> Optional[str]:
    return s.lower() if isinstance(s, str) else s

# ---------- JSON-RPC handler ----------

async def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON-RPC 2.0 handler supporting:
      - system.health
      - analytics.presence.log
      - analytics.presence.get
      - analytics.kpi.get
      - analytics.kpi.ingest
    """
    mid = req.get("id")
    method = req.get("method")
    p = (req.get("params") or {}) if isinstance(req.get("params"), dict) else {}

    # ---- system.health ----
    if method == "system.health":
        return _ok(mid, {"service": SERVICE_NAME, "version": API_VERSION, "db": _db_path(), "status": "ok"})

    conn = await _ensure_db()

    # ---- analytics.presence.log ----
    if method == "analytics.presence.log":
        missing = _require(p, ["query", "engine"])
        if missing:
            return _err(mid, *missing)

        # Optional idempotence granularity: 'second' (default) | 'minute' | 'day'
        gran = (p.get("ts_floor") or "second").lower()
        now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        provided = _parse_iso_utc(p.get("ts"))
        ts_final = _floor_time(provided or now, gran)

        seen_val = 1 if bool(p.get("seen")) else 0

        citations = p.get("citations")
        if citations is not None:
            try:
                citations = int(citations)
            except Exception:
                return _err(mid, "invalid_params", "citations must be an integer")

        position = p.get("position")
        if position is not None:
            try:
                position = int(position)
            except Exception:
                return _err(mid, "invalid_params", "position must be an integer")

        definition_version = p.get("definition_version") or os.getenv("PRESENCE_DEFINITION_VERSION")

        # Normalized enums (best-effort)
        engine = _norm_engine(p.get("engine"))
        serp_surface = _norm_surface(p.get("serp_surface"))
        engine_flavor = _norm_flavor(p.get("engine_flavor"))

        # Try insert; if duplicate per UNIQUE, update fields for idempotence
        try:
            cur = await conn.execute(
                """INSERT OR IGNORE INTO presence
                   (query, engine, geo, serp_surface, engine_flavor, seen, citations, details, ts, position, definition_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    p["query"],
                    engine,
                    p.get("geo"),
                    serp_surface,
                    engine_flavor,
                    seen_val,
                    citations,
                    p.get("details"),
                    _iso_utc(ts_final),
                    position,
                    definition_version,
                ),
            )
            await conn.commit()

            if cur.rowcount == 0:
                # Duplicate → update selected fields (seen/citations/details/position/definition_version)
                await conn.execute(
                    """UPDATE presence
                       SET seen = COALESCE(?, seen),
                           citations = COALESCE(?, citations),
                           details = COALESCE(?, details),
                           position = COALESCE(?, position),
                           definition_version = COALESCE(?, definition_version)
                     WHERE query = ? AND engine = ? AND (geo IS ? OR geo = ?) AND ts = ?""",
                    (
                        seen_val,
                        citations,
                        p.get("details"),
                        position,
                        definition_version,
                        p["query"],
                        engine,
                        p.get("geo"),
                        p.get("geo"),
                        _iso_utc(ts_final),
                    ),
                )
                await conn.commit()

            return _ok(mid, {"ok": True, "ts": _iso_utc(ts_final)})

        except Exception as e:
            return _err(mid, "db_error", "Failed to upsert presence", {"detail": str(e)})

    # ---- analytics.presence.get ----
    if method == "analytics.presence.get":
        qs: List[str] = p.get("queries") or []
        frm = p.get("from")
        to = p.get("to")
        engine = p.get("engine")
        geo = p.get("geo")
        serp_surface = p.get("serp_surface")
        engine_flavor = p.get("engine_flavor")
        limit = p.get("limit")

        # normalize filters
        engine = _norm_engine(engine)
        serp_surface = _norm_surface(serp_surface)
        engine_flavor = _norm_flavor(engine_flavor)

        try:
            if limit is not None:
                limit = int(limit)
                if limit <= 0:
                    return _err(mid, "invalid_params", "limit must be > 0")
        except Exception:
            return _err(mid, "invalid_params", "limit must be an integer")

        sql = """
SELECT query, engine, geo, serp_surface, engine_flavor, seen, citations, details, ts, position, definition_version
FROM presence
WHERE 1=1
"""
        args: List[Any] = []
        if qs:
            sql += " AND query IN (%s)" % ",".join(["?"] * len(qs))
            args += qs
        if engine:
            sql += " AND engine = ?"
            args.append(engine)
        if geo:
            sql += " AND geo = ?"
            args.append(geo)
        if serp_surface:
            sql += " AND serp_surface = ?"
            args.append(serp_surface)
        if engine_flavor:
            sql += " AND engine_flavor = ?"
            args.append(engine_flavor)
        if frm:
            sql += " AND ts >= ?"
            args.append(frm)
        if to:
            sql += " AND ts <= ?"
            args.append(to)
        sql += " ORDER BY ts ASC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)

        try:
            cur = await conn.execute(sql, args)
            rows = [dict(r) for r in await cur.fetchall()]
            return _ok(mid, {"presence": rows})
        except Exception as e:
            return _err(mid, "db_error", "Failed to query presence", {"detail": str(e)})

    # ---- analytics.kpi.get ----
    if method == "analytics.kpi.get":
        metrics: List[str] = p.get("metrics") or []
        if not metrics:
            return _err(mid, "invalid_params", "metrics is required and must be a non-empty array")

        frm = p.get("from")
        to = p.get("to")
        geo = p.get("geo")
        src = p.get("src")  # str | List[str] | None
        include_src = bool(p.get("include_src", False))

        series: List[Dict[str, Any]] = []
        try:
            for m in metrics:
                sql = "SELECT t, v" + (", src" if include_src else "") + " FROM kpi_points WHERE metric = ?"
                args: List[Any] = [m]

                if geo:
                    sql += " AND geo = ?"
                    args.append(geo)

                if isinstance(src, list) and src:
                    sql += " AND src IN (%s)" % ",".join(["?"] * len(src))
                    args += src
                elif isinstance(src, str) and src:
                    sql += " AND src = ?"
                    args.append(src)

                if frm:
                    sql += " AND t >= ?"
                    args.append(frm)
                if to:
                    sql += " AND t <= ?"
                    args.append(to)

                sql += " ORDER BY t ASC"

                cur = await conn.execute(sql, args)
                rows = await cur.fetchall()
                if include_src:
                    pts = [{"t": r["t"], "v": r["v"], "src": r["src"]} for r in rows]
                else:
                    pts = [{"t": r["t"], "v": r["v"]} for r in rows]

                series.append({"metric": m, "points": pts})

            return _ok(mid, {"series": series})
        except Exception as e:
            return _err(mid, "db_error", "Failed to query KPI series", {"detail": str(e)})

    # ---- analytics.kpi.ingest ----
    if method == "analytics.kpi.ingest":
        """
        params: {
          points: [{metric, t, v, geo?, src?}, ...]
        }
        Upserts by (metric, t, geo, src).
        """
        pts: List[Dict[str, Any]] = p.get("points") or []
        if not pts or not isinstance(pts, list):
            return _err(mid, "invalid_params", "points must be a non-empty array")

        # Validate and build batches
        batch: List[Tuple[Any, ...]] = []
        for row in pts:
            if not isinstance(row, dict):
                return _err(mid, "invalid_params", "each point must be an object")
            for k in ("metric", "t", "v"):
                if row.get(k) is None:
                    return _err(mid, "invalid_params", f"point missing required field: {k}")

            metric = str(row["metric"])
            t = str(row["t"])  # ISO-8601 string for lexical ordering
            try:
                v = float(row["v"])
            except Exception:
                return _err(mid, "invalid_params", "v must be a number")

            geo_val = row.get("geo")
            src_val = row.get("src") or "csv"

            batch.append((metric, t, v, geo_val, src_val))

        try:
            # UPSERT on (metric, t, geo, src)
            await conn.executemany(
                """
INSERT INTO kpi_points (metric, t, v, geo, src)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(metric, t, geo, src) DO UPDATE SET v = excluded.v
""",
                batch,
            )
            await conn.commit()
            return _ok(mid, {"ok": True, "upserted": len(batch)})
        except Exception as e:
            return _err(mid, "db_error", "Failed to ingest KPI points", {"detail": str(e)})

    return _err(mid, "method_not_found", str(method))
