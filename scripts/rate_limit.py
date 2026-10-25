"""
scripts/rate_limit.py

Small in-memory sliding-window rate limiter. Enough for a single backend
instance; a multi-instance deployment would need a shared store (e.g. Redis).
"""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit, window_seconds):
        self.limit = limit
        self.window = window_seconds
        self.hits = defaultdict(deque)  # key -> timestamps of recent requests
        self.lock = threading.Lock()

    def check(self, key):
        """Record a request for `key`. Returns None if allowed, else seconds to wait."""
        now = time.monotonic()
        with self.lock:
            timestamps = self.hits[key]
            while timestamps and now - timestamps[0] >= self.window:
                timestamps.popleft()

            if len(timestamps) >= self.limit:
                if not timestamps:  # limit of 0: always blocked
                    return max(1, int(self.window))
                return max(1, int(self.window - (now - timestamps[0])) + 1)

            timestamps.append(now)

            # Drop idle keys so the dict doesn't grow with every IP ever seen
            if len(self.hits) > 10_000:
                for k in [k for k, ts in self.hits.items() if not ts]:
                    del self.hits[k]
            return None

    def reset(self):
        with self.lock:
            self.hits.clear()


def client_ip(request):
    # Render sits behind a proxy, so the real client is the first X-Forwarded-For
    # entry. A client can spoof this header, which is why a global cap also applies.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
