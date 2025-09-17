# dp_agent/run_demo_single_process.py
import os, io, time, torch, json
from cryptography.fernet import Fernet
from dp_agent.dp_agent import DPAgent

# 👇 centralized receipts
from ..centralised_receipts import CentralReceiptManager


def make_state_dict():
    return {"w1": torch.randn(20, 20), "b1": torch.randn(20)}


def diff_privacy():
    # 🔹 Load or generate Fernet key (shared across pipeline)
    os.makedirs("keys", exist_ok=True)
    key_path = "keys/fernet.key"
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            demo_key = f.read().strip()
    else:
        demo_key = Fernet.generate_key()
        with open(key_path, "wb") as f:
            f.write(demo_key)
        print("Generated new Fernet key -> keys/fernet.key")

    # 🔹 Trainer creates encrypted update
    os.makedirs("secure_store/local_updates", exist_ok=True)
    fname = f"trainer_{int(time.time() * 1000)}.pt.enc"
    path = os.path.join("secure_store/local_updates", fname)

    sd = make_state_dict()
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw = buf.getvalue()

    f = Fernet(demo_key)
    enc = f.encrypt(raw)

    with open(path, "wb") as wf:
        wf.write(enc)

    # 🔹 Create trainer receipt using CentralReceiptManager
    rm = CentralReceiptManager()
    trainer_receipt = rm.create_receipt(
        agent="trainer-agent",
        session_id=f"sess-{int(time.time())}",
        operation="train_step",
        params={
            "epochs": 1,
            "batch_size": 32,
            "dataset_size": 1000,
        },
        outputs=["file://" + path],
    )

    os.makedirs("receipts", exist_ok=True)
    trainer_receipt_uri = rm.write_receipt(trainer_receipt, out_dir="receipts")

    print("Trainer receipt created:", trainer_receipt_uri)

    # 🔹 Run DP agent with same Fernet key
    dp = DPAgent(
        clip_norm=1.0,
        noise_multiplier=1.2,
        secure_store_dir="secure_store/local_updates",
        receipts_dir="receipts",
        fernet_key=demo_key,   # same Fernet key
    )

    dp_result = dp.process_local_update(
        trainer_receipt["outputs"][0], metadata=trainer_receipt
    )
    print("DP receipt created:", dp_result["receipt_uri"])


if __name__ == "__main__":
    diff_privacy()
