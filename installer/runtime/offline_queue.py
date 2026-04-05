"""
offline_queue.py — Phase 9: persistent offline queue for failed uploads.

When a SubmitReceipt RPC fails due to network issues, the receipt is saved
to disk. On the next daemon cycle, queued receipts are retried before
processing new data, ensuring no updates are silently lost.

Queue layout:
    ~/.federated/state/offline_queue/
        <uuid>.json        ← serialised Receipt fields
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_QUEUE_DIR = Path.home() / ".federated" / "state" / "offline_queue"
_MAX_QUEUE = 50        # cap — oldest entries dropped beyond this
_MAX_RETRIES = 10      # per-entry retry ceiling


def _ensure_dir():
    _QUEUE_DIR.mkdir(parents=True, exist_ok=True)


# ── Public API ────────────────────────────────────────────────────────────────

def enqueue(receipt_fields: dict) -> str:
    """
    Persist a failed receipt to the offline queue.
    Returns the queue entry ID (filename stem).
    """
    _ensure_dir()

    # Enforce cap: drop oldest entry if full
    entries = sorted(_QUEUE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    while len(entries) >= _MAX_QUEUE:
        oldest = entries.pop(0)
        log.warning("[offline_queue] Queue full — dropping oldest entry: %s", oldest.name)
        oldest.unlink(missing_ok=True)

    entry_id = uuid.uuid4().hex
    entry = {
        "id": entry_id,
        "retries": 0,
        "receipt": receipt_fields,
    }
    path = _QUEUE_DIR / f"{entry_id}.json"
    path.write_text(json.dumps(entry, indent=2))
    log.info("[offline_queue] Enqueued receipt %s (queue size now %d)", entry_id, len(entries) + 1)
    return entry_id


def drain(stub, call_with_retry_fn, Receipt) -> int:
    """
    Retry all queued receipts.  Removes entries on success or when retry
    ceiling is exceeded.  Returns the number of successfully submitted entries.
    """
    _ensure_dir()
    entries = sorted(_QUEUE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not entries:
        return 0

    log.info("[offline_queue] Draining %d queued receipts", len(entries))
    success_count = 0

    for path in entries:
        try:
            entry = json.loads(path.read_text())
        except Exception as e:
            log.warning("[offline_queue] Corrupt entry %s — removing: %s", path.name, e)
            path.unlink(missing_ok=True)
            continue

        retries = entry.get("retries", 0)
        if retries >= _MAX_RETRIES:
            log.error(
                "[offline_queue] Entry %s exceeded %d retries — dropping",
                entry["id"], _MAX_RETRIES
            )
            path.unlink(missing_ok=True)
            continue

        rf = entry["receipt"]
        try:
            receipt_msg = Receipt(
                device_id=bytes.fromhex(rf["device_id_hex"]),
                round_id=rf["round_id"],
                payload_hash=bytes.fromhex(rf["payload_hash_hex"]),
                epsilon_spent=rf["epsilon_spent"],
                signature=bytes.fromhex(rf["signature_hex"]),
                enc_uri=rf["enc_uri"],
                scheme=rf["scheme"],
                nonce=rf.get("nonce", ""),
            )
            ack = call_with_retry_fn(stub.SubmitReceipt, receipt_msg, timeout=15)
            if ack.ok:
                log.info("[offline_queue] Queued receipt %s submitted successfully", entry["id"])
                path.unlink(missing_ok=True)
                success_count += 1
            else:
                log.warning("[offline_queue] Server rejected queued receipt %s", entry["id"])
                _increment_retry(path, entry)
        except Exception as e:
            log.warning("[offline_queue] Retry failed for %s: %s", entry["id"], e)
            _increment_retry(path, entry)

    return success_count


def queue_size() -> int:
    """Return the number of pending queued receipts."""
    _ensure_dir()
    return len(list(_QUEUE_DIR.glob("*.json")))


def _increment_retry(path: Path, entry: dict):
    entry["retries"] = entry.get("retries", 0) + 1
    try:
        path.write_text(json.dumps(entry, indent=2))
    except Exception as e:
        log.warning("[offline_queue] Failed to update retry count for %s: %s", path.name, e)


# ── Helper: build serialisable receipt dict from a Receipt protobuf message ──

def receipt_to_dict(receipt_msg) -> dict:
    """Convert a Receipt protobuf to a JSON-serialisable dict for the queue."""
    return {
        "device_id_hex":    receipt_msg.device_id.hex(),
        "round_id":         receipt_msg.round_id,
        "payload_hash_hex": receipt_msg.payload_hash.hex(),
        "epsilon_spent":    receipt_msg.epsilon_spent,
        "signature_hex":    receipt_msg.signature.hex(),
        "enc_uri":          receipt_msg.enc_uri,
        "scheme":           receipt_msg.scheme,
        "nonce":            receipt_msg.nonce,
    }