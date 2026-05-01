"""
Supabase compliance logger
==========================

Sends non-compliance events (no-helmet, improper-footwear) to a
Supabase REST endpoint in the background.  Designed to never block the
inference / tracker / GUI threads:

* All POSTs run on a single daemon worker thread.
* If the network is down or the service rejects the row, the event is
  dropped after a brief retry — we never raise into the caller.
* Duplicate events for the same (motorcycle_track_id, violation_type)
  inside ``DEDUP_WINDOW_S`` seconds are suppressed.

Expected Supabase table schema (create once, manually, in the dashboard)::

    create table public.compliance_logs (
        id              bigserial primary key,
        created_at      timestamptz not null default now(),
        violation_type  text not null,
        motorcycle_id   integer,
        rider_id        integer,
        rider_count     integer,
        confidence      real,
        source          text,
        notes           text
    );

The PostgREST ``anon`` role must have ``insert`` privilege.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

# ── public config (filled in by configure()) ──────────────────────────────────

_SUPABASE_URL: str = ""
_SUPABASE_KEY: str = ""
_TABLE: str = "compliance_logs"
_HAS_CREDENTIALS: bool = False
# Default OFF — DB logging is opt-in via the sidebar checkbox.
_USER_ENABLED: bool = False

DEDUP_WINDOW_S: float = 8.0
"""Suppress repeat events for the same (moto_id, type) inside this window."""

REQUEST_TIMEOUT_S: float = 3.0


@dataclass
class ViolationEvent:
    violation_type: str
    motorcycle_id: Optional[int] = None
    rider_id: Optional[int] = None
    rider_count: Optional[int] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    notes: Optional[str] = None
    video_timestamp: Optional[float] = None

# ── module state ──────────────────────────────────────────────────────────────

_q: "queue.Queue[ViolationEvent]" = queue.Queue(maxsize=512)
_worker: Optional[threading.Thread] = None
_stop = threading.Event()
_recent_lock = threading.Lock()
_recent: Dict[str, float] = {}
_status_cb: Optional[Callable[[str, str], None]] = None
"""Optional callback (level, message) used to surface logger activity in
the GUI's console panel.  ``level`` is one of "info" / "error"."""


def configure(url: str, anon_key: str,
              table: str = "compliance_logs",
              status_cb: Optional[Callable[[str, str], None]] = None) -> None:
    """Initialise (or reconfigure) the Supabase connection."""
    global _SUPABASE_URL, _SUPABASE_KEY, _TABLE, _HAS_CREDENTIALS, _status_cb
    _SUPABASE_URL = (url or "").rstrip("/")
    _SUPABASE_KEY = anon_key or ""
    _TABLE = table
    _HAS_CREDENTIALS = bool(_SUPABASE_URL and _SUPABASE_KEY)
    _status_cb = status_cb
    _ensure_worker()


def is_enabled() -> bool:
    return _HAS_CREDENTIALS and _USER_ENABLED


def has_credentials() -> bool:
    return _HAS_CREDENTIALS


def set_enabled(enabled: bool) -> None:
    """User-facing on/off toggle for DB logging.

    When credentials are missing this still records the user's preference
    but ``is_enabled()`` will keep returning False.
    """
    global _USER_ENABLED
    _USER_ENABLED = bool(enabled)


def shutdown(timeout: float = 1.0) -> None:
    _stop.set()
    if _worker is not None:
        _worker.join(timeout=timeout)


# ── public submit ─────────────────────────────────────────────────────────────

def submit(event: ViolationEvent) -> bool:
    """Queue an event for upload.  Returns False if deduped or disabled."""
    if not is_enabled():
        return False

    key = f"{event.violation_type}:{event.motorcycle_id}:{event.rider_id}"
    now = time.monotonic()
    with _recent_lock:
        last = _recent.get(key, 0.0)
        if now - last < DEDUP_WINDOW_S:
            return False
        _recent[key] = now
        # GC old entries occasionally
        if len(_recent) > 1024:
            _recent_purge(now)

    try:
        _q.put_nowait(event)
        return True
    except queue.Full:
        return False


# ── internals ─────────────────────────────────────────────────────────────────

def _recent_purge(now: float) -> None:
    cutoff = now - DEDUP_WINDOW_S * 4
    for k in [k for k, t in _recent.items() if t < cutoff]:
        _recent.pop(k, None)


def _ensure_worker() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    _stop.clear()
    _worker = threading.Thread(
        target=_worker_loop, name="SupabaseLogger", daemon=True)
    _worker.start()


def _worker_loop() -> None:
    while not _stop.is_set():
        try:
            ev = _q.get(timeout=0.5)
        except queue.Empty:
            continue
        if not is_enabled():
            continue
        ok, detail = _post_event(ev)
        if not ok and _status_cb is not None:
            try:
                label = ev.violation_type.upper()
                _status_cb(
                    "error",
                    f"Supabase insert failed ({label}): {detail}",
                )
            except Exception:
                pass


def _post_event(ev: ViolationEvent) -> tuple[bool, str]:
    """Returns (success, detail_string)."""
    payload = {
        "violation_type": ev.violation_type,
        "motorcycle_id": ev.motorcycle_id,
        "rider_id": ev.rider_id,
        "rider_count": ev.rider_count,
        "confidence": ev.confidence,
        "source": ev.source,
        "notes": ev.notes,
        "video_timestamp": ev.video_timestamp,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    body = json.dumps([payload]).encode("utf-8")
    url = f"{_SUPABASE_URL}/rest/v1/{_TABLE}"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("apikey", _SUPABASE_KEY)
    req.add_header("Authorization", f"Bearer {_SUPABASE_KEY}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Prefer", "return=minimal")
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                if 200 <= resp.status < 300:
                    return True, "ok"
                return False, f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read(512).decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            detail = f"HTTP {e.code} {err_body[:200]}"
            # 4xx → bad payload, table missing, or RLS blocked; don't retry
            if 400 <= e.code < 500:
                return False, detail
            # 5xx → server error; retry once
            if attempt == 0:
                time.sleep(0.4)
                continue
            return False, detail
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            detail = str(e)
            if attempt == 0:
                time.sleep(0.4)
                continue
            return False, detail
    return False, "unknown error"


# ── helper to build dict from event for GUI display ───────────────────────────

def event_to_text(ev: ViolationEvent) -> str:
    bits = [ev.violation_type.upper()]
    if ev.motorcycle_id is not None:
        bits.append(f"moto#{ev.motorcycle_id}")
    if ev.rider_id is not None:
        bits.append(f"rider#{ev.rider_id}")
    if ev.rider_count is not None:
        bits.append(f"riders={ev.rider_count}")
    if ev.confidence is not None:
        bits.append(f"conf={ev.confidence:.2f}")
    return " ".join(bits)


def to_dict(ev: ViolationEvent) -> Dict[str, Any]:
    return {
        "violation_type": ev.violation_type,
        "motorcycle_id": ev.motorcycle_id,
        "rider_id": ev.rider_id,
        "rider_count": ev.rider_count,
        "confidence": ev.confidence,
        "source": ev.source,
        "notes": ev.notes,
    }
