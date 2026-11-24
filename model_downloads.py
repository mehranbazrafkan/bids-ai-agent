"""Model downloads: check what is local, fetch what is not, report progress.

Models load from the Hugging Face cache on first use. Until they are
there, a request silently downloads a multi-gigabyte checkpoint with no
visible progress, and a desktop app has nothing to show for the wait.
This module makes the cache inspectable and the download a first-class
operation:

* :func:`is_downloaded` — whether a model is already in the local cache
  (``GET /models`` reports this per entry, so the desktop app can flag
  what is ready to use offline);
* :func:`start` — download a model in a background thread;
* :func:`status` — how far that download has got, in bytes.

The cache layout is the standard huggingface_hub one —
``<cache>/models--<org>--<name>/{refs,snapshots,blobs}`` — so a model
copied into the folder by hand (or fetched by any HF tool) counts as
downloaded, and everything downloaded here is usable by any HF tool.
The cache directory comes from the environment the same way
huggingface_hub resolves it: ``HF_HUB_CACHE``, then ``HF_HOME``, then
the default under the user's home.

Progress is real, not decorative: the total is the sum of the repo's
file sizes (asked of the Hub once, before the download starts), and the
downloaded count is the size of what is actually on disk under the
model's ``blobs`` directory, sampled while the download runs. When the
total cannot be established (offline metadata, an unknown repo), it
stays ``None`` and the client shows an indeterminate bar with the bytes
counted so far.

Everything here is stdlib plus ``huggingface_hub``, which the agent's
environment already carries for model loading. One download at a time:
a second :func:`start` while one is running answers with the current
status rather than racing it.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:                                                   # pragma: no cover
    from huggingface_hub import constants as hf_constants
    from huggingface_hub import model_info, snapshot_download
    HAS_HF = True
except Exception:                                      # pragma: no cover
    hf_constants = None                                # type: ignore[assignment]
    HAS_HF = False

#: How often the on-disk byte count is sampled while a download runs.
POLL_SECONDS = 0.5

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "name": "",
    "status": "idle",          # idle | downloading | done | error
    "downloaded_bytes": 0,
    "total_bytes": None,
    "error": "",
    "started": 0.0,
    "finished": 0.0,
}
_THREAD: Optional[threading.Thread] = None


# ---------------------------------------------------------------------------
# The cache directory
# ---------------------------------------------------------------------------

def cache_dir() -> Path:
    """The Hub cache this process reads and writes, as huggingface_hub
    resolves it: ``HF_HUB_CACHE`` wins, then ``HF_HOME``/``hub``, then
    the default under the user's home."""
    if HAS_HF and hf_constants is not None:
        return Path(hf_constants.HF_HUB_CACHE)
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if env:
        base = Path(env)
        return base if base.name == "hub" or "HF_HUB_CACHE" in os.environ \
            else base / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_dir(name: str, root: Path | None = None) -> Path:
    """The cache folder for *name*, in the standard ``models--org--repo``
    layout. Slash and separator are both ``-`` pairs in that spelling, so
    ``Qwen/Qwen3-1.7B`` becomes ``models--Qwen--Qwen3-1.7B``."""
    safe = name.strip("/").replace("/", "--")
    return (root or cache_dir()) / f"models--{safe}"


# ---------------------------------------------------------------------------
# What is already local
# ---------------------------------------------------------------------------

def is_downloaded(name: str, root: Path | None = None) -> bool:
    """Whether *name* is usable from the local cache right now.

    The honest-enough check: the repo folder exists, ``refs/main``
    points at a revision, and that revision's snapshot holds at least
    one file. A half-fetched download (an interrupted run leaves
    ``*.incomplete`` blobs behind) does not count as ready.
    """
    repo = _repo_dir(name, root)
    refs = repo / "refs"
    snapshots = repo / "snapshots"
    if not refs.is_dir() or not snapshots.is_dir():
        return False
    try:
        revision = next(
            (p for p in refs.iterdir() if p.is_file() and p.read_bytes().strip()),
            None,
        )
    except OSError:
        return False
    if revision is None:
        return False
    head = snapshots / revision.read_text(encoding="utf-8", errors="replace").strip()
    if not head.is_dir():
        return False
    try:
        if not any(head.rglob("*")):
            return False
        blobs = repo / "blobs"
        if blobs.is_dir() and any(blobs.glob("*.incomplete")):
            return False
    except OSError:
        return False
    return True


def _dir_size(directory: Path) -> int:
    """Bytes of every regular file under *directory*; 0 when unreadable."""
    total = 0
    try:
        for path in directory.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def downloaded_bytes(name: str, root: Path | None = None) -> int:
    """How much of *name* is on disk: the ``blobs`` directory holds the
    real bytes (snapshots only symlink into it), so that is the part
    that grows while a download runs."""
    return _dir_size(_repo_dir(name, root) / "blobs")


# ---------------------------------------------------------------------------
# What the repo needs
# ---------------------------------------------------------------------------

def total_bytes(name: str) -> Optional[int]:
    """The repo's total download size in bytes, or ``None`` when it
    cannot be known (offline, unknown repo). Asked of the Hub once per
    download, before it starts."""
    if not HAS_HF:
        return None
    try:
        info = model_info(name, files_metadata=True)
    except Exception:                                  # noqa: BLE001
        return None
    sizes = [
        int(s.size or 0)
        for s in (getattr(info, "siblings", None) or [])
        if getattr(s, "size", None)
    ]
    return sum(sizes) if sizes else None


# ---------------------------------------------------------------------------
# The download itself
# ---------------------------------------------------------------------------

def _publish(**fields: Any) -> None:
    with _LOCK:
        _STATE.update(fields)


def status() -> Dict[str, Any]:
    """A snapshot of the current (or last) download. Copy, not the live
    dict: the client polls this while the worker mutates it."""
    with _LOCK:
        return dict(_STATE)


def _worker(name: str) -> None:
    """The background download: total first, then the fetch, with a
    sampler thread keeping the on-disk count fresh while it runs."""
    total = total_bytes(name)
    _publish(
        name=name, status="downloading", total_bytes=total,
        downloaded_bytes=downloaded_bytes(name), error="",
        started=time.time(), finished=0.0,
    )

    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(POLL_SECONDS):
            _publish(downloaded_bytes=downloaded_bytes(name))

    sampler = threading.Thread(target=sample, name="bids-agent-dl-progress",
                               daemon=True)
    sampler.start()
    try:
        if not HAS_HF:
            raise RuntimeError("huggingface_hub is not installed")
        snapshot_download(name)
    except Exception as exc:                           # noqa: BLE001
        _publish(
            status="error", error=f"{type(exc).__name__}: {exc}",
            finished=time.time(),
        )
    else:
        _publish(
            status="done", downloaded_bytes=downloaded_bytes(name),
            finished=time.time(),
        )
        if total is not None:
            _publish(downloaded_bytes=max(downloaded_bytes(name), 0))
    finally:
        stop.set()


def start(name: str) -> Dict[str, Any]:
    """Download *name* in the background; return the status immediately.

    Never blocks and never raises for a busy or repeated request: a
    download already running (or already finished for this repo) simply
    answers with its status, and the client polls :func:`status` for
    progress. A blank name is the caller's bug and raises.
    """
    name = str(name or "").strip()
    if not name:
        raise ValueError("a model name is required")
    global _THREAD
    with _LOCK:
        running = _STATE["status"] == "downloading"
        same = _STATE["name"] == name
    if running and same:
        return status()
    if running:
        # One at a time: a second repo while one is fetching would race
        # the single status slot. The client sees the active download.
        return status()
    if is_downloaded(name):
        # Already local: there is nothing to fetch. Record it as done so
        # the client's poll sees a finished job rather than a stale one.
        _publish(
            name=name, status="done", error="",
            downloaded_bytes=downloaded_bytes(name),
        )
        return status()
    _publish(name=name, status="downloading", downloaded_bytes=0,
             total_bytes=None, error="")
    _THREAD = threading.Thread(
        target=_worker, args=(name,), name="bids-agent-download", daemon=True,
    )
    _THREAD.start()
    return status()


__all__ = ["cache_dir", "downloaded_bytes", "is_downloaded", "start", "status",
           "total_bytes"]
