import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

import httpx

from coldsnap.config import DOWNLOAD_CONCURRENCY

USER_AGENT = "coldsnap/0.1 research"
RETRY_STATUS = {429, 500, 502, 503, 504}


class Stats:
    def __init__(self):
        self.bytes = 0
        self.requests = 0
        self.started = time.perf_counter()

    def as_dict(self) -> dict:
        return {
            "bytes": self.bytes,
            "requests": self.requests,
            "wall_seconds": round(time.perf_counter() - self.started, 1),
        }


def client(http2: bool = True, timeout: float = 60.0) -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=DOWNLOAD_CONCURRENCY, max_keepalive_connections=DOWNLOAD_CONCURRENCY
    )
    return httpx.AsyncClient(
        http2=http2,
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


async def get(
    http: httpx.AsyncClient,
    url: str,
    stats: Stats | None = None,
    byte_range: tuple[int, int | None] | None = None,
    retries: int = 6,
) -> httpx.Response | None:
    headers = {}
    if byte_range is not None:
        start, end = byte_range
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    for attempt in range(retries):
        try:
            response = await http.get(url, headers=headers)
        except (httpx.TransportError, httpx.TimeoutException):
            response = None
        if response is not None and response.status_code not in RETRY_STATUS:
            if stats is not None:
                stats.requests += 1
                stats.bytes += len(response.content)
            return response
        await asyncio.sleep(min(60.0, 0.5 * 2**attempt) * (0.5 + random.random()))
    return None


def get_sync(url: str, timeout: float = 120.0) -> httpx.Response:
    with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as http:
        for attempt in range(6):
            try:
                response = http.get(url)
                if response.status_code not in RETRY_STATUS:
                    return response
            except (httpx.TransportError, httpx.TimeoutException):
                pass
            time.sleep(min(60.0, 0.5 * 2**attempt))
    raise RuntimeError(f"failed to fetch {url}")


def download(url: str, path: Path, timeout: float = 600.0) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    response = get_sync(url, timeout=timeout)
    response.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(response.content)
    tmp.rename(path)
    return path


async def gather_limited(jobs: Iterable[Callable[[], Awaitable]], limit: int = DOWNLOAD_CONCURRENCY) -> list:
    semaphore = asyncio.Semaphore(limit)

    async def run(job):
        async with semaphore:
            return await job()

    return await asyncio.gather(*(run(job) for job in jobs))


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.done: set[str] = set()
        if path.exists():
            self.done = {line.strip() for line in path.read_text().splitlines() if line.strip()}
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a")

    def __contains__(self, key: str) -> bool:
        return key in self.done

    def add(self, key: str) -> None:
        if key not in self.done:
            self.done.add(key)
            self._handle.write(key + "\n")
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
