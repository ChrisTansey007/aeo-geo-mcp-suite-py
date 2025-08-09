import time
class TokenBucket:
    def __init__(self, cap: int = 60, refill_per_sec: int = 60):
        self.cap = cap; self.tokens = cap
        self.refill = 1.0 / refill_per_sec; self.last = time.time()
    def take(self) -> bool:
        now = time.time(); elapsed = now - self.last
        add = int(elapsed / self.refill)
        if add>0:
            self.tokens = min(self.cap, self.tokens + add); self.last = now
        if self.tokens>0:
            self.tokens -= 1; return True
        return False
