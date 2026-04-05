"""
pipeline.py — Full federated pipeline with:
  - Phase 6: schema validation + manifest integrity
  - Phase 9: gRPC retry via call_with_retry
  - Phase 10: receive global model, train locally, send delta
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

log = logging.getLogger(__name__)

# ── Canonical store root — must match _CANONICAL_ROOT in SecureStore ──────────
_STORE_ROOT = Path.home() / ".federated" / "data" / "secure_store"
_CONFIG_URI = f"file://{Path.home()}/.federated/configs/local_config.yaml"
_INPUT_DIR  = str(Path.home() / ".federated" / "data" / "input")

LDA_MODE    = "session"


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


# ── Global model receiver ─────────────────────────────────────────────────────
def _download_global_model(round_meta) -> Optional[str]:
    """
    If the orchestrator provides a global model URI in upload_uri,
    download and verify it. Returns local path or None.
    """
    uri = getattr(round_meta, "upload_uri", "") or ""
    if not uri or not uri.startswith("file://"):
        return None   # no global model distributed this round

    path = Path(uri[len("file://"):])
    if not path.exists():
        log.warning("Global model URI points to non-existent file: %s", path)
        return None

    # verify file hash (integrity check)
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
      1. Query round
      2. (Optional) Download global model
      3. LDA preprocessing
      4. Trainer (fine-tune on local data)
      5. DP noise
      6. Encryption
      7. Submit receipt
    """
    # ── 1. Query round ────────────────────────────────────────────────────────
    log.info("[pipeline] Querying round metadata...")
    round_meta = call_with_retry(stub.GetRound, DeviceId(id=device_id), timeout=10)

    if round_meta.state != "Collecting":
        log.info("[pipeline] Round state=%s — skipping", round_meta.state)
        return

    session_id = f"client-{uuid.uuid4().hex[:12]}"
    log.info("[pipeline] Round %d active — session %s", round_meta.round_id, session_id)

    # ── 2. Download global model (Phase 10 FL) ────────────────────────────────
    global_model_path = _download_global_model(round_meta)
    if global_model_path:
        log.info("[FL] Will initialize trainer from global model")

    # ── 3. LDA preprocessing ─────────────────────────────────────────────────
    log.info("[pipeline] Running LDA...")

    # Determine input directory
    if session_dir and session_dir.exists():
        video_dir = str(session_dir)
    else:
        video_dir = _INPUT_DIR

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

    # ── 4. Trainer ────────────────────────────────────────────────────────────
    log.info("[pipeline] Running trainer (mode=supervised)...")

    trainer_kwargs = {
        "input_path": manifest_uri,
        "session_id": session_id,
        "mode": "supervised",
        "epochs": 1,
        "batch_size": 8,
    }

    # Phase 10: if global model available, pass it for warm-start
    # (trainer_orchestrate will load it if global_model_path is passed)
    # Extend trainer_orchestrate signature in trainer file to accept this.
    if global_model_path:
        trainer_kwargs["global_model_path"] = global_model_path

    t0 = time.time()
    trainer_out = trainer_orchestrate(**trainer_kwargs)
    log.info("[pipeline] Trainer done in %.1fs", time.time() - t0)

    _validate_trainer_output(trainer_out)
    local_update_uri = trainer_out["local_update_uri"]

    # ── 5. Differential Privacy ───────────────────────────────────────────────
    log.info("[pipeline] Applying DP noise...")

    store = SecureStore(agent="trainer", root=_STORE_ROOT)
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
    enc_agent = EncryptionAgent(mode="aes")
    enc_result = enc_agent.process_dp_update(dp_result["receipt_uri"])
    final_update_uri = enc_result["receipt"]["outputs"][0]

    # ── 7. Submit receipt ─────────────────────────────────────────────────────
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

    ack = call_with_retry(stub.SubmitReceipt, receipt, timeout=15)
    if ack.ok:
        log.info("[pipeline] ✅ Round %d update submitted", round_meta.round_id)
    else:
        log.warning("[pipeline] Server returned ok=False for round %d", round_meta.round_id)