# MCP Suite (monorepo)

Servers: mcp-crawl, mcp-search, mcp-schema, mcp-content-lint, mcp-publisher, mcp-analytics.
Transports: STDIO/HTTP. Endpoints: POST /rpc, GET /metrics, GET /healthz.

Quickstart:
```bash
pnpm i
pnpm -r build
pnpm --filter @mcp/mcp-crawl start:http
```
