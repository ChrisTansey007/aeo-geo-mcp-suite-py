import os

def check_api_key(provided: str | None) -> bool:
    key = os.getenv("MCP_API_KEY")
    if not key:
        return True
    return (provided or "").replace("Bearer ", "").strip() == key
