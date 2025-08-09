#!/usr/bin/env python3
import argparse, asyncio, os
from packages.mcp_common.transport_http import make_app
from packages.mcp_common.stdio import run_stdio
from .service import handle

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["http","stdio"], default="http")
    p.add_argument("--port", type=int, default=3001)
    args = p.parse_args()
    if args.transport == "http":
        import uvicorn
        app = make_app("mcp-content-lint", handle)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        asyncio.run(run_stdio(handle))

if __name__ == "__main__":
    main()
