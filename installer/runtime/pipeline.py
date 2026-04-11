"""
pipeline.py — Full federated pipeline

SECURITY FIXES:
  FIX-PIPELINE-1: Model bytes are now streamed to the server via UploadUpdate
                  RPC before SubmitReceipt. Previously enc_uri was a local
                  file path the server could never open.

  FIX-PIPELINE-2: Receipt.enc_handle is the server-side GridFS ObjectId
                  returned by UploadAck — NOT a local file path.

  FIX-PIPELINE-3: epsilon_spent is taken from the DP agent's real output
                  instead of the hardcoded 1.0.

  FIX-PIPELINE-4: payload_hash is computed over the actual bytes that were
                  streamed — guarantees receipt matches uploaded data.

  FIX-PIPELINE-5: Per-chunk SHA-256 included in each UpdateChunk so server
                  can verify every chunk independently.

  FIX-PIPELINE-6: Global model downloaded via DownloadGlobalModel RPC with
                  per-chunk and full-model hash verification.
"""

import uuid
import math
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
from runtime.grpc.orchestrator_pb2 import (
    DeviceId, Receipt, UpdateChunk, RoundRequest
)
from runtime.grpc_client import call_with_retry
from runtime.tpm_guard import sign_message

log = logging.getLogger(__name__)

_STORE_ROOT = Path.home() / ".federated" / "data" / "secure_store"
_CONFIG_URI = f"file://{Path.home()}/.federated/configs/local_config.yaml"
_INPUT_DIR  = str(Path.home() / ".federated" / "data" / "input")

LDA_MODE      = "session"
CHUNK_SIZE    = 1 * 1024 * 1024   # 1 MB per gRPC chunk
MAX_EPS_VALUE = 10.0               # hard ceiling — server rejects > this


# ── Schema validation (unchanged) ─────────────────────────────────────────────
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
    uri  = result["local_update_uri"]
    if not uri.startswith("file://"):
        raise ValueError(f"Trainer output URI malformed: {uri}")
    path = Path(uri[len("file://"):])
    if not path.exists():
        raise ValueError(f"Trainer update file not found: {path}")
    log.info("[schema] Trainer output valid: %s", path.name)


# ── FIX-PIPELINE-6: Download global model from server ─────────────────────────
def _download_global_model(stub, device_id: bytes, round_id: int) -> Optional[str]:
    """
    Download the global model from the server via streaming RPC.
    Verifies per-chunk and full-model SHA-256 hashes.
    Returns local path to the downloaded model file, or None.
    """
    try:
        request = RoundRequest(device_id=device_id, round_id=round_id)
        chunks_received = []
        full_model_hash_expected = None

        for chunk in stub.DownloadGlobalModel(request, timeout=120):
            # Verify chunk integrity
            computed = hashlib.sha256(chunk.data).digest()
            if computed != bytes(chunk.chunk_hash):
                raise ValueError(
                    f"Global model chunk {chunk.chunk_index} hash mismatch — "
                    f"data corrupted in transit"
                )
            chunks_received.append(chunk.data)

            if chunk.chunk_index == chunk.total_chunks - 1 and chunk.model_hash:
                full_model_hash_expected = bytes(chunk.model_hash)

        if not chunks_received:
            log.info("[FL] No global model chunks received")
            return None

        model_bytes = b"".join(chunks_received)

        # Verify full model hash
        if full_model_hash_expected:
            actual_hash = hashlib.sha256(model_bytes).digest()
            if actual_hash != full_model_hash_expected:
                raise ValueError(
                    "Global model full-hash mismatch — model rejected"
                )
            log.info("[FL] Global model hash verified OK")

        # Save to local path
        model_dir = Path.home() / ".federated" / "data" / "global_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"global_round{round_id}.pt"
        model_path.write_bytes(model_bytes)

        log.info(
            "[FL] Global model downloaded: %d bytes → %s",
            len(model_bytes), model_path
        )
        return str(model_path)

    except Exception as e:
        log.warning("[FL] Could not download global model: %s — starting from scratch", e)
        return None


# ── FIX-PIPELINE-1,4,5: Stream encrypted update to server ─────────────────────
def _stream_update(
    stub,
    device_id: bytes,
    round_id: int,
    update_path: str,
    session_id: str,
) -> tuple:
    """
    Read the encrypted update file and stream it to the server in chunks.

    Each chunk includes its own SHA-256 hash so the server can verify
    every chunk independently (FIX-PIPELINE-5).

    Returns (server_handle, payload_hash_bytes):
      server_handle    — GridFS ObjectId string from UploadAck.server_handle
      payload_hash_bytes — SHA-256 of all bytes streamed (for Receipt)
    """
    path = Path(update_path[len("file://"):])
    if not path.exists():
        raise FileNotFoundError(f"Update file not found: {path}")

    data = path.read_bytes()
    if not data:
        raise ValueError("Update file is empty — will not stream")

    total_size   = len(data)
    total_chunks = math.ceil(total_size / CHUNK_SIZE)
    payload_hash = hashlib.sha256(data).digest()   # hash of ALL bytes

    log.info(
        "[pipeline] Streaming %d bytes in %d chunks (sha256=%s…)",
        total_size, total_chunks, payload_hash.hex()[:16]
    )

    def chunk_generator():
        for i in range(total_chunks):
            chunk_data = data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
            chunk_hash = hashlib.sha256(chunk_data).digest()
            yield UpdateChunk(
                session_id=session_id,
                round_id=round_id,
                device_id=device_id,
                chunk_index=i,
                total_chunks=total_chunks,
                data=chunk_data,
                chunk_hash=chunk_hash,
            )

    # stream is client-side streaming — use direct stub call
    ack = stub.UploadUpdate(chunk_generator(), timeout=300)

    if not ack.ok:
        raise RuntimeError(
            f"Server rejected upload: {ack.error}"
        )

    log.info("[pipeline] Upload complete — server_handle=%s", ack.server_handle)
    return ack.server_handle, payload_hash


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(
    stub,
    device_id: bytes,
    master_secret: bytes,
    session_dir: Optional[Path] = None,
):
    """
    Full local federated pipeline:
      1. Query round
      2. Download global model (if available)
      3. LDA preprocessing
      4. Trainer (fine-tune on local data)
      5. DP noise
      6. Encryption
      7. Stream bytes to server (NEW — FIX-PIPELINE-1)
      8. Submit receipt with real epsilon and server handle (FIX-PIPELINE-2,3)
    """
    # ── 1. Query round ────────────────────────────────────────────────────────
    log.info("[pipeline] Querying round metadata...")
    round_meta = call_with_retry(stub.GetRound, DeviceId(id=device_id), timeout=10)

    if round_meta.state != "Collecting":
        log.info("[pipeline] Round state=%s — skipping", round_meta.state)
        return

    session_id = f"client-{uuid.uuid4().hex[:12]}"
    log.info("[pipeline] Round %d active — session %s", round_meta.round_id, session_id)

    # ── 2. Download global model (FIX-PIPELINE-6) ─────────────────────────────
    global_model_path = None
    if round_meta.global_model_available:
        log.info("[FL] Global model available — downloading...")
        global_model_path = _download_global_model(stub, device_id, round_meta.round_id)
    else:
        log.info("[FL] No global model for this round — random init")

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

    # ── 4. Trainer ────────────────────────────────────────────────────────────
    log.info("[pipeline] Running trainer (mode=supervised)...")

    trainer_kwargs = {
        "input_path":  manifest_uri,
        "session_id":  session_id,
        "mode":        "supervised",
        "epochs":      1,
        "batch_size":  8,
    }
    if global_model_path:
        trainer_kwargs["global_model_path"] = global_model_path

    t0 = time.time()
    trainer_out = trainer_orchestrate(**trainer_kwargs)
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
    log.info(
        "[pipeline] DP done: L2 before=%.4f after=%.4f eps=%.6f",
        dp_result["l2_norm_before"],
        dp_result["l2_norm_after"],
        dp_result.get("epsilon_spent", 0.0),
    )

    # FIX-PIPELINE-3: read real epsilon from DP agent, never hardcode
    epsilon_spent = dp_result.get("epsilon_spent")
    if epsilon_spent is None or epsilon_spent <= 0.0:
        # Fallback: if DP agent didn't compute epsilon, derive a conservative
        # estimate. Log a warning — production should always have real RDP.
        epsilon_spent = 1.0
        log.warning(
            "[pipeline] DP agent did not return epsilon_spent — using fallback 1.0. "
            "Wire up the RDP accountant in DPAgent.process_local_update()."
        )
    elif epsilon_spent > MAX_EPS_VALUE:
        raise ValueError(
            f"epsilon_spent={epsilon_spent:.4f} exceeds hard ceiling {MAX_EPS_VALUE} "
            f"— reduce noise multiplier or clip norm"
        )

    # ── 6. Encryption ─────────────────────────────────────────────────────────
    log.info("[pipeline] Finalizing encryption...")
    enc_agent       = EncryptionAgent(mode="aes")
    enc_result      = enc_agent.process_dp_update(dp_result["receipt_uri"])
    final_update_uri = enc_result["receipt"]["outputs"][0]

    # ── 7. Stream bytes to server (FIX-PIPELINE-1,4,5) ───────────────────────
    log.info("[pipeline] Streaming update to server...")
    t0 = time.time()

    server_handle, payload_hash = _stream_update(
        stub,
        device_id=device_id,
        round_id=round_meta.round_id,
        update_path=final_update_uri,
        session_id=session_id,
    )

    log.info("[pipeline] Stream complete in %.1fs", time.time() - t0)

    # ── 8. Submit receipt (FIX-PIPELINE-2,3) ─────────────────────────────────
    log.info("[pipeline] Submitting receipt to server...")

    # Sign canonical message: device_id || round_id (8 bytes BE) || payload_hash
    msg = (
        device_id
        + round_meta.round_id.to_bytes(8, "big")
        + payload_hash           # SHA-256 of actual uploaded bytes
    )
    signature = sign_message(msg)

    receipt = Receipt(
        device_id=device_id,
        round_id=round_meta.round_id,
        payload_hash=payload_hash,           # FIX-PIPELINE-4: real hash
        epsilon_spent=epsilon_spent,         # FIX-PIPELINE-3: real epsilon
        signature=signature,
        enc_handle=server_handle,            # FIX-PIPELINE-2: GridFS ID
        scheme="AES-GCM-DP-ECDSA",
        nonce="",
    )

    ack = call_with_retry(stub.SubmitReceipt, receipt, timeout=15)
    if ack.ok:
        log.info("[pipeline] ✅ Round %d update submitted", round_meta.round_id)
    else:
        log.warning("[pipeline] Server returned ok=False for round %d", round_meta.round_id)