"""
offline_queue.py — Persistent offline queue for failed uploads.

SECURITY FIX:
  FIX-QUEUE-1: Queue entries are now encrypted using AES-GCM via SecureStore
               before being written to disk. Previously all queued receipts
               were stored as plaintext JSON in state/offline_queue/*.json,
               exposing device_id, round_id, payload_hash, epsilon, enc_handle,
               and signatures to anyone with filesystem access.

Queue layout:
    ~/.federated/state/offline_queue/
        <uuid>.json.enc   ← AES-GCM encrypted JSON
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from core.centralized_secure_store import SecureStore

log = logging.getLogger(__name__)

_QUEUE_DIR  = Path.home() / ".federated" / "state" / "offline_queue"
_STORE_ROOT = Path.home() / ".federated" / "data" / "secure_store"
_MAX_QUEUE  = 50
_MAX_RETRIES = 10


def _store() -> SecureStore:
    """Return a SecureStore instance for the offline queue."""
    return SecureStore(agent="offline-queue", root=_STORE_ROOT)


def _queue_uri(entry_id: str) -> str:
    path = _QUEUE_DIR / f"{entry_id}.json.enc"
    return f"file://{path}"


def _ensure_dir():
    _QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def enqueue(receipt_fields: dict) -> str:
    """
    Persist a failed receipt to the offline queue (encrypted).
    Returns the queue entry ID.
    """
    _ensure_dir()

    store = _store()

    # Enforce cap: drop oldest entry
    entries = sorted(_QUEUE_DIR.glob("*.json.enc"), key=lambda p: p.stat().st_mtime)
    while len(entries) >= _MAX_QUEUE:
        oldest = entries.pop(0)
        log.warning("[offline_queue] Queue full — dropping oldest: %s", oldest.name)
        oldest.unlink(missing_ok=True)

    entry_id = uuid.uuid4().hex
    entry = {
        "id":      entry_id,
        "retries": 0,
        "receipt": receipt_fields,
    }

    uri = _queue_uri(entry_id)
    # FIX-QUEUE-1: encrypt before writing
    store.encrypt_write(uri, json.dumps(entry).encode())

    log.info(
        "[offline_queue] Enqueued receipt %s (queue size %d)",
        entry_id, len(entries) + 1
    )
    return entry_id


def drain(stub, call_with_retry_fn, Receipt) -> int:
    """
    Retry all queued receipts.
    Returns the number of successfully submitted entries.
    """
    _ensure_dir()

    store   = _store()
    entries = sorted(_QUEUE_DIR.glob("*.json.enc"), key=lambda p: p.stat().st_mtime)
    if not entries:
        return 0

    log.info("[offline_queue] Draining %d queued receipts", len(entries))
    success_count = 0

    for path in entries:
        entry_id = path.stem.replace(".json", "")
        uri      = f"file://{path}"

        try:
            # FIX-QUEUE-1: decrypt before use
            raw   = store.decrypt_read(uri)
            entry = json.loads(raw.decode())
        except Exception as e:
            log.warning("[offline_queue] Corrupt/unreadable entry %s — removing: %s",
                        path.name, e)
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
                enc_handle=rf["enc_handle"],   # FIX: was enc_uri (local path)
                scheme=rf["scheme"],
                nonce=rf.get("nonce", ""),
            )
            ack = call_with_retry_fn(stub.SubmitReceipt, receipt_msg, timeout=15)
            if ack.ok:
                log.info(
                    "[offline_queue] Queued receipt %s submitted successfully",
                    entry["id"]
                )
                path.unlink(missing_ok=True)
                success_count += 1
            else:
                log.warning("[offline_queue] Server rejected queued receipt %s", entry["id"])
                _increment_retry(store, path, uri, entry)
        except Exception as e:
            log.warning("[offline_queue] Retry failed for %s: %s", entry["id"], e)
            _increment_retry(store, path, uri, entry)

    return success_count


def queue_size() -> int:
    _ensure_dir()
    return len(list(_QUEUE_DIR.glob("*.json.enc")))


def _increment_retry(store: SecureStore, path: Path, uri: str, entry: dict):
    entry["retries"] = entry.get("retries", 0) + 1
    try:
        path.unlink(missing_ok=True)
        store.encrypt_write(uri, json.dumps(entry).encode())
    except Exception as e:
        log.warning("[offline_queue] Failed to update retry count: %s", e)


def receipt_to_dict(receipt_msg) -> dict:
    """Convert a Receipt protobuf to a JSON-serialisable dict for the queue."""
    return {
        "device_id_hex":    receipt_msg.device_id.hex(),
        "round_id":         receipt_msg.round_id,
        "payload_hash_hex": receipt_msg.payload_hash.hex(),
        "epsilon_spent":    receipt_msg.epsilon_spent,
        "signature_hex":    receipt_msg.signature.hex(),
        "enc_handle":       receipt_msg.enc_handle,   # GridFS ObjectId
        "scheme":           receipt_msg.scheme,
        "nonce":            receipt_msg.nonce,
    }