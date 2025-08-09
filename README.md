# AEO/GEO MCP Suite (Python)


**GEO = Generative Engine Optimization**

Standalone MCP servers (STDIO + HTTP) with AEO/GEO features:

## Data directory

* `packages/local/` holds all local market & NAP data (aliases, NAP truth, etc).
* `packages/geo/` is now a stub for backward compatibility and will be removed in a future release.

## Environment variables

New (preferred):


Old (deprecated, supported for one release):


## Migration

* Replace `GEO_ALIASES_PATH` → `LOCAL_ALIASES_PATH`
* Replace `GEO_MARKET` → `LOCAL_MARKET`
* Replace `GEO_NAP_FILE` → `LOCAL_NAP_FILE`
* Replace `DEFAULT_GEO_KEY` → `DEFAULT_LOCAL_KEY`

All code now prefers `LOCAL_*` envs and paths, but will fall back to `GEO_*` for one minor release. Update your configs and integrations accordingly.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start HTTP transports (examples)
python servers/mcp-crawl/server.py  --transport=http --port=3001
python servers/mcp-search/server.py --transport=http --port=3031
python servers/mcp-schema/server.py --transport=http --port=3041
python servers/mcp-content-lint/server.py --transport=http --port=3052
python servers/mcp-publisher/server.py --transport=http --port=3051 --repo ./content-repo --base-url https://example.com
python servers/mcp-analytics/server.py --transport=http --port=3061 --db ./analytics.db

# Or STDIO (good for MCP-capable clients)
python servers/mcp-crawl/server.py  --transport=stdio
```

All HTTP servers expose `/metrics` (Prometheus); set `MCP_API_KEY` to require Bearer auth.
python servers/mcp-crawl/server.py  --transport=http --port=3001
