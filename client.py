import os
import io
import time
import json
import torch
import argparse
import pyarrow.parquet as pq
import numpy as np
import pyarrow as pa

# --- LDA ---
from LDA.app.main import preprocess, PreprocessRequest

# --- Trainer ---
from trainer_agent.trainer import train_model
from centralized_secure_store import SecureStore
from centralised_receipts import CentralReceiptManager

# --- DP Agent ---
from dp_agent.dp_agent import DPAgent

# --- Encryption Agent ---
from enc_agent.enc_agent import EncryptionAgent


# -------------------
# Helper: Read embeddings from parquet
# -------------------
import io
import pyarrow.parquet as pq

def read_embeddings_from_parquet(store: SecureStore, uri: str):
    """
    Decrypt parquet bytes from SecureStore into PyArrow Table.
    """
    data = store.decrypt_read(uri)
    buf = io.BytesIO(data)
    return pq.read_table(buf)


def read_manifest(store: SecureStore, manifest_uri: str):
    """
    Decrypts the manifest JSONL file and returns a list of parquet URIs.
    """
    data = store.decrypt_read(manifest_uri)
    lines = data.decode().splitlines()
    manifest = [json.loads(line) for line in lines]

    parquet_uris = sorted(set(m["uri"] for m in manifest if m["uri"].endswith(".parquet.enc")))
    return parquet_uris

# -------------------
# Main pipeline
# -------------------
def run_pipeline(args):
    # Central managers
    store = SecureStore("./secure_store")
    rm = CentralReceiptManager()

    # ---------------- LDA Step ----------------
    print("\n=== STEP 1: LDA Preprocessing ===")
    lda_req = PreprocessRequest(
        mode=args.lda_mode,
        inputs={args.input_type: args.input_path},
        config_uri="file://configs/local_config.yaml",
    )
    lda_result = preprocess(lda_req)
    print("LDA outputs:", lda_result)

    session_parquet = lda_result["artifact_manifest"]

    # ---------------- Trainer Step ----------------
    print("\n=== STEP 2: Training Agent ===")

    # 1. Read manifest → get parquet URIs
    parquet_uris = read_manifest(store, session_parquet)
    if not parquet_uris:
        raise RuntimeError("No parquet embeddings found in manifest")

    # 2. Load all parquet tables into one tensor
    tables = [read_embeddings_from_parquet(store, uri) for uri in parquet_uris]
    combined = pa.concat_tables(tables)
    df = combined.to_pandas()

    # Assume embeddings are stored in a column called "embedding" (list of floats)
    if "embedding" in df.columns:
        embs = np.stack(df["embedding"].to_numpy())
    else:
        # fallback: use all numeric columns directly
        embs = df.select_dtypes(include=[np.number]).to_numpy()

    if embs.shape[0] == 0:
        raise RuntimeError("Embeddings are empty after loading parquet")

    X = torch.tensor(embs, dtype=torch.float32)
    y = torch.zeros(X.shape[0], dtype=torch.long)  # dummy labels

    delta, _ = train_model(
        X, y, input_dim=X.shape[1],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )

    # Save local update via SecureStore
    update_fname = f"trainer_{int(time.time() * 1000)}.pt.enc"
    update_path = os.path.join("secure_store/local_updates", update_fname)
    os.makedirs(os.path.dirname(update_path), exist_ok=True)

    buf = io.BytesIO()
    torch.save(delta, buf)
    buf.seek(0)
    store.encrypt_write("file://" + update_path, buf.getvalue())

    trainer_receipt = rm.create_receipt(
        agent="trainer-agent",
        session_id=lda_result["session_id"],
        operation="train_step",
        params={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
        },
        outputs=["file://" + update_path],
    )
    trainer_receipt_uri = rm.write_receipt(trainer_receipt, out_dir="receipts")
    print("Trainer receipt:", trainer_receipt_uri)

    # ---------------- DP Agent Step ----------------
    print("\n=== STEP 3: Differential Privacy Agent ===")
    dp = DPAgent(
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        secure_store_dir="secure_store/local_updates",
        receipts_dir="receipts",
    )
    dp_result = dp.process_local_update(
        trainer_receipt["outputs"][0], metadata=trainer_receipt
    )
    print("DP receipt:", dp_result["receipt_uri"])

    # ---------------- Encryption Agent Step ----------------
    print("\n=== STEP 4: Encryption Agent ===")
    enc = EncryptionAgent(mode=args.enc_mode)
    enc_result = enc.process_dp_update(dp_result["receipt_uri"])
    print("Encryption receipt:", enc_result["receipt_uri"])

    print("\n=== PIPELINE COMPLETE ===")
    return {
        "lda": lda_result,
        "trainer": trainer_receipt,
        "dp": dp_result["receipt"],
        "enc": enc_result["receipt"],
    }


# -------------------
# CLI entrypoint
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline")

    # LDA options
    parser.add_argument("--lda-mode", default="text",
                        choices=["text", "session", "batch", "continuous"],
                        help="Mode for LDA preprocessing")
    parser.add_argument("--input-type", default="text_dir",
                        choices=["text_dir", "video_dir", "audio_dir"],
                        help="Which input modality directory to use")
    parser.add_argument("--input-path", default="./sample_texts",
                        help="Path to input directory")

    # Trainer options
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")

    # DP Agent options
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.2)

    # Encryption Agent options
    parser.add_argument("--enc-mode", default="aes",
                        choices=["aes", "fernet", "kms_envelope", "he_ckks", "smpc"],
                        help="Encryption scheme to apply after DP")

    args = parser.parse_args()

    result = run_pipeline(args)
    print("\nFinal pipeline outputs:")
    print(json.dumps(result, indent=2))
