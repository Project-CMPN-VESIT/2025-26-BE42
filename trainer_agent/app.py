# trainer_agent/main.py
import torch, io, uuid, time
from pathlib import Path
from typing import Dict, Any

from trainer import train_model
from utils import read_embeddings_from_parquet
from security.secure_store import SecureStore
from ..centralised_receipts import CentralReceiptManager

# SecureStore (same as LDA)
store = SecureStore("./secure_store")

# Centralized receipt manager
receipt_mgr = CentralReceiptManager(agent="trainer-agent")


def train(
    session_parquet: str,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Train on embeddings from parquet and produce:
      - Encrypted local update (secure_store/local_updates/*.pt.enc)
      - Signed receipt (via CentralReceiptManager)
    """
    # 1) Read embeddings
    embs = read_embeddings_from_parquet(session_parquet)
    if len(embs) == 0:
        raise RuntimeError("No embeddings found in parquet")

    X = torch.tensor(embs, dtype=torch.float32)
    y = torch.zeros(X.shape[0], dtype=torch.long)  # dummy labels

    # 2) Train model -> delta update
    delta, _ = train_model(
        X, y, input_dim=X.shape[1],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    )

    # 3) Save encrypted update
    tmp = io.BytesIO()
    torch.save(delta, tmp)
    tmp.seek(0)

    update_fname = f"{uuid.uuid4().hex}.pt.enc"
    update_uri = f"file://secure_store/local_updates/{update_fname}"

    store.encrypt_write(update_uri, tmp.getvalue())

    # 4) Create signed receipt
    receipt = receipt_mgr.create_receipt(
        session_id=f"sess-{int(time.time())}",
        operation="train",
        params={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": device,
            "session_parquet": session_parquet,
        },
        outputs=[update_uri]
    )

    return {
        "local_update_uri": update_uri,
        "receipt": receipt
    }
