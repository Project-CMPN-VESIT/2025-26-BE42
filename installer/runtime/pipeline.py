"""
pipeline.py — Full federated pipeline

Phase 9:  Offline queue — failed SubmitReceipt calls are persisted and
          retried on the next daemon cycle.
Phase 10: Global model is now actually loaded in the trainer (was passed
          via **kwargs but never consumed — fixed by adding explicit param
          to orchestrate() call).
"""

import uuid
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

from agents.lda.main import preprocess, PreprocessRequest
from agents.trainer.trainer_mentalbert_privacy import orchestrate as trainer_orchestrate
from agents.dp.dp_agent import DPAgent
from agents.enc.enc_agent import EncryptionAgent
from core.centralized_secure_store import SecureStore
from runtime.grpc.orchestrator_pb2 import DeviceId, Receipt
from runtime.grpc_client import call_with_retry
from runtime.tpm_guard import sign_message
from runtime import offline_queue          # Phase 9

log = logging.getLogger(__name__)

_STORE_ROOT = Path.home() / ".federated" / "data" / "secure_store"
_CONFIG_URI = f"file://{Path.home()}/.federated/configs/local_config.yaml"
_INPUT_DIR  = str(Path.home() / ".federated" / "data" / "input")

LDA_MODE = "session"

# ── Schema validation ─────────────────────────────────────────────────────────

_REQUIRED_MANIFEST_KEYS = {"session_id", "artifact_manifest", "receipts", "count"}

def _validate_lda_output(result: dict):
    missing = _REQUIRED_MANIFEST_KEYS - set(result.keys())
    if missing:
        raise ValueError(f"LDA output missing keys: {missing}")
    if result.get("count", 0) == 0:
        raise ValueError("LDA produced 0 rows — check input data")
    log.info("[schema] LDA output valid: %d rows", result["count"])


_REQUIRED_TRAINER_KEYS = {"local_update_uri"}

def _validate_trainer_output(result: dict):
    missing = _REQUIRED_TRAINER_KEYS - set(result.keys())
    if missing:
        raise ValueError(f"Trainer output missing keys: {missing}")
    uri = result["local_update_uri"]
    if not uri.startswith("file://"):
        raise ValueError(f"Trainer output URI malformed: {uri}")
    path = Path(uri[len("file://"):])
    if not path.exists():
        raise ValueError(f"Trainer update file not found: {path}")
    log.info("[schema] Trainer output valid: %s", path.name)


# ── Global model receiver (Phase 10) ─────────────────────────────────────────

def _download_global_model(round_meta) -> Optional[str]:
    uri = getattr(round_meta, "upload_uri", "") or ""
    if not uri or not uri.startswith("file://"):
        return None

    path = Path(uri[len("file://"):])
    if not path.exists():
        log.warning("[FL] Global model URI points to non-existent file: %s", path)
        return None

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    log.info("[FL] Global model received: %s (sha256=%s…)", path.name, digest[:16])
    return str(path)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    stub,
    device_id: bytes,
    master_secret: bytes,
    session_dir: Optional[Path] = None,
):
    """
    Full local federated pipeline:
      0. Drain offline queue (Phase 9)
      1. Query round
      2. Download global model (Phase 10)
      3. LDA preprocessing
      4. Trainer (fine-tune; warm-start from global model if available)
      5. Differential Privacy noise
      6. Encryption
      7. Submit receipt (with offline queue fallback — Phase 9)
    """

    # ── 0. Drain any previously failed receipts ───────────────────────────────
    queued = offline_queue.queue_size()
    if queued > 0:
        log.info("[pipeline] Draining %d queued receipts before new round", queued)
        drained = offline_queue.drain(stub, call_with_retry, Receipt)
        log.info("[pipeline] Drained %d/%d queued receipts", drained, queued)

    # ── 1. Query round ────────────────────────────────────────────────────────
    log.info("[pipeline] Querying round metadata...")
    round_meta = call_with_retry(stub.GetRound, DeviceId(id=device_id), timeout=10)

    if round_meta.state != "Collecting":
        log.info("[pipeline] Round state=%s — skipping", round_meta.state)
        return

    session_id = f"client-{uuid.uuid4().hex[:12]}"
    log.info("[pipeline] Round %d active — session %s", round_meta.round_id, session_id)

    # ── 2. Download global model (Phase 10) ───────────────────────────────────
    global_model_path = _download_global_model(round_meta)
    if global_model_path:
        log.info("[FL] Will warm-start trainer from global model: %s", global_model_path)

    # ── 3. LDA preprocessing ─────────────────────────────────────────────────
    log.info("[pipeline] Running LDA...")

    video_dir = str(session_dir) if (session_dir and session_dir.exists()) else _INPUT_DIR

    lda_req = PreprocessRequest(
        mode=LDA_MODE,
        inputs={"video_dir": video_dir},
        config_uri=_CONFIG_URI,
    )

    t0 = time.time()
    lda_result = preprocess(lda_req)
    log.info("[pipeline] LDA done in %.1fs", time.time() - t0)

    _validate_lda_output(lda_result)
    manifest_uri = lda_result["artifact_manifest"]

    # ── 4. Trainer (Phase 10: global_model_path is now explicit kwarg) ────────
    log.info("[pipeline] Running trainer (mode=supervised)...")

    # Phase 10 FIX: global_model_path passed explicitly so orchestrate() can
    # actually load it — previously it arrived via **kwargs and was ignored.
    t0 = time.time()
    trainer_out = trainer_orchestrate(
        input_path=manifest_uri,
        session_id=session_id,
        mode="supervised",
        epochs=1,
        batch_size=8,
        global_model_path=global_model_path,   # ← Phase 10: now consumed
    )
    log.info("[pipeline] Trainer done in %.1fs", time.time() - t0)

    _validate_trainer_output(trainer_out)
    local_update_uri = trainer_out["local_update_uri"]

    # ── 5. Differential Privacy ───────────────────────────────────────────────
    log.info("[pipeline] Applying DP noise...")

    store    = SecureStore(agent="trainer", root=_STORE_ROOT)
    dp_agent = DPAgent(
        clip_norm=1.0,
        noise_multiplier=1.0,
        mechanism="gaussian",
        store=store,
    )

    dp_result = dp_agent.process_local_update(
        local_update_uri,
        session_id=session_id,
        metadata={"session_id": session_id},
    )
    log.info("[pipeline] DP done: L2 before=%.4f after=%.4f",
             dp_result["l2_norm_before"], dp_result["l2_norm_after"])

    # ── 6. Encryption ─────────────────────────────────────────────────────────
    log.info("[pipeline] Finalizing encryption...")
    enc_agent      = EncryptionAgent(mode="aes")
    enc_result     = enc_agent.process_dp_update(dp_result["receipt_uri"])
    final_update_uri = enc_result["receipt"]["outputs"][0]

    # ── 7. Submit receipt (Phase 9: offline queue on failure) ─────────────────
    log.info("[pipeline] Submitting receipt to server...")
    payload_hash = hashlib.sha256(final_update_uri.encode()).digest()
    msg = (
        device_id
        + round_meta.round_id.to_bytes(8, "big")
        + payload_hash
    )
    signature = sign_message(msg)

    receipt = Receipt(
        device_id=device_id,
        round_id=round_meta.round_id,
        payload_hash=payload_hash,
        epsilon_spent=1.0,
        signature=signature,
        enc_uri=final_update_uri,
        scheme="AES-GCM-SecureStore",
        nonce="",
    )

    try:
        ack = call_with_retry(stub.SubmitReceipt, receipt, timeout=15)
        if ack.ok:
            log.info("[pipeline] ✅ Round %d update submitted", round_meta.round_id)
        else:
            log.warning("[pipeline] Server returned ok=False — queueing for retry")
            offline_queue.enqueue(offline_queue.receipt_to_dict(receipt))
    except Exception as e:
        log.error("[pipeline] SubmitReceipt failed: %s — saving to offline queue", e)
        offline_queue.enqueue(offline_queue.receipt_to_dict(receipt))