import sys, json, asyncio

async def run_stdio(handler):
    for line in sys.stdin:
        line=line.strip()
        if not line: continue
        try:
            payload = json.loads(line)
        except Exception:
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":None,"error":{"code":"bad_json","message":"invalid JSON"}})+"\n"); sys.stdout.flush(); continue
        res = await handler(payload)
        sys.stdout.write(json.dumps(res)+"\n"); sys.stdout.flush()
