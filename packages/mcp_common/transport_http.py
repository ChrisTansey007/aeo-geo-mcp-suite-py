import json, os
from fastapi import FastAPI, Request, Response, Header
from fastapi.responses import StreamingResponse, PlainTextResponse
from .auth import check_api_key
from .metrics import REQS, ERRS, DUR, export
from contextlib import contextmanager
import time

@contextmanager
def duration(server: str, method: str):
    start = time.time()
    try:
        yield
    finally:
        DUR.labels(server, method).observe(time.time() - start)

def make_app(server_name: str, handler):
    app = FastAPI()
    @app.post("/")
    async def rpc(req: Request, authorization: str | None = Header(default=None)):
        body = await req.body()
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return Response(status_code=400, content=json.dumps({"error":{"code":"bad_json","message":"invalid JSON"}}))
        if not check_api_key(authorization):
            return Response(status_code=401, content=json.dumps({"error":{"code":"unauthorized","message":"bad api key"}}))
        method = payload.get("method","unknown")
        REQS.labels(server_name, method).inc()
        with duration(server_name, method):
            try:
                res = await handler(payload)
            except Exception as e:
                ERRS.labels(server_name, method, getattr(e,'code','internal')).inc()
                return Response(status_code=200, content=json.dumps({"jsonrpc":"2.0","id":payload.get("id"),"error":{"code":"internal","message":str(e)}}))
        return Response(status_code=200, content=json.dumps(res))
    @app.get("/metrics")
    async def metrics():
        data, ctype = export()
        return Response(status_code=200, media_type=ctype, content=data)
    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "server": server_name}
    return app
