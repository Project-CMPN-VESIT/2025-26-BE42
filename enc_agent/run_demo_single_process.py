# enc_agent/run_demo_single_process.py
import os
from cryptography.fernet import Fernet

from dp_agent.run_demo_single_process import main as run_dp_demo
from enc_agent.enc_agent import EncryptionAgent


def encrypt_agent():
    # 🔹 Run DP demo first (produces dp_receipt in receipts/)
    run_dp_demo()

    # 🔹 Load shared Fernet key (same as DP + decrypt_agent)
    key_path = "keys/fernet.key"
    if not os.path.exists(key_path):
        raise FileNotFoundError("Fernet key not found! Run dp_agent first to generate one.")
    with open(key_path, "rb") as f:
        fernet_key = f.read().strip()

    # 🔹 Find the latest DP receipt
    receipts_dir = "receipts"
    files = [f for f in os.listdir(receipts_dir) if f.startswith("dp_") and f.endswith(".json")]
    if not files:
        print("No DP receipts found!")
        return
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(receipts_dir, f)))
    dp_receipt_path = os.path.join(receipts_dir, latest)

    print(f"Encrypting DP update from: {dp_receipt_path}")

    # 🔹 Use Fernet mode (consistent with DP Agent demo)
    enc_agent = EncryptionAgent(mode="fernet", symmetric_key=fernet_key)
    result = enc_agent.process_dp_update(dp_receipt_path)

    receipt = result["receipt"]
    receipt_uri = result["receipt_uri"]

    print("✅ Encryption receipt created and stored:")
    print("   Receipt URI:", receipt_uri)
    print("   Receipt JSON:", receipt)

