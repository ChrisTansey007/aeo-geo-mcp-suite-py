from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

registry = CollectorRegistry()
REQS = Counter('mcp_requests_total','requests', ['server','method'], registry=registry)
ERRS = Counter('mcp_errors_total','errors', ['server','method','code'], registry=registry)
DUR = Histogram('mcp_request_seconds','duration', ['server','method'], registry=registry)

def metrics_app():
    def app(scope, receive, send):
        # minimal ASGI for /metrics mounted under FastAPI route
        pass
    return app

def export():
    return generate_latest(registry), CONTENT_TYPE_LATEST
