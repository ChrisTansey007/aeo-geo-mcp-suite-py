import hashlib

def stable_block_id(xpath: str, payload: str, order: int) -> str:
    h = hashlib.sha1((xpath + "|" + payload).encode("utf-8")).hexdigest()[:4]
    return f"b-{order:04d}-{h}"
