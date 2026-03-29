# ~/.federated/runtime/pipeline.py

import uuid
import hashlib
from pathlib import Path

from agents.lda.main import preprocess, PreprocessRequest
from agents.trainer.trainer_mentalbert_privacy import orchestrate as trainer_orchestrate
from agents.dp.dp_agent import DPAgent
from agents.enc.enc_agent import EncryptionAgent
from core.centralized_secure_store import SecureStore
from runtime.grpc.orchestrator_pb2 import DeviceId, Receipt
from runtime.tpm_guard import sign_message

# --------------------------------------------------
# Runtime pipeline (NO security / NO gRPC setup here)
# --------------------------------------------------

LDA_MODE = "session"
CONFIG_URI = f"file://{Path.home()}/.federated/configs/local_config.yaml"
VIDEO_DIR = str(Path.home() / ".federated" / "data" / "input")  # installer may override later


def run_pipeline(stub, device_id: bytes, master_secret: bytes):
    """
    Executes full local pipeline and uploads encrypted update.
    Assumes:
      - runtime_guard already passed
      - mTLS gRPC stub provided
    """

    # --------------------------------------------------
    # 1) Query round
    # --------------------------------------------------
    round_meta = stub.GetRound(DeviceId(id=device_id), timeout=10)

    if round_meta.state != "Collecting":
        return

    session_id = f"client-{uuid.uuid4().hex}"

    # ==================================================
    # 2) LDA
    # ==================================================
    lda_req = PreprocessRequest(
        mode=LDA_MODE,
        inputs={"video_dir": VIDEO_DIR},
        config_uri=CONFIG_URI,
        session_id=session_id,
    )
    lda_result = preprocess(lda_req)

    manifest_uri = lda_result["artifact_manifest"]

    # ==================================================
    # 3) Trainer
    # ==================================================
    trainer_out = trainer_orchestrate(
        input_path=manifest_uri,
        session_id=session_id,
        mode="supervised",
        epochs=1,
        batch_size=8,
    )

    local_update_uri = trainer_out["local_update_uri"]

    # ==================================================
    # 4) Differential Privacy
    # ==================================================
    store = SecureStore(
        agent="trainer-agent",
        root=Path.home() / ".federated" / "secure_store"
    )

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

    # ==================================================
    # 5) Encryption
    # ==================================================
    enc_agent = EncryptionAgent(mode="aes")
    enc_result = enc_agent.process_dp_update(dp_result["receipt_uri"])

    final_update_uri = enc_result["receipt"]["outputs"][0]

    # ==================================================
    # 6) Submit receipt
    # ==================================================
    payload_hash = hashlib.sha256(
        final_update_uri.encode("utf-8")
    ).digest()

    msg = (
        device_id +
        round_meta.round_id.to_bytes(8, "big") +
        payload_hash
    )

    # NOTE:
    # signature is produced by TPM-backed key in grpc_client
    signature = sign_message(msg)

    receipt = Receipt(
        device_id=device_id,
        round_id=round_meta.round_id,
        payload_hash=payload_hash,
        epsilon_spent=1.0,
        signature=signature,
        enc_uri=final_update_uri,
        scheme="AES-GCM",
        nonce="",
    )

    stub.SubmitReceipt(receipt, timeout=10)
