import json
import base64
from pathlib import Path
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# --- CONFIG ---
enc_file = "secure_store/sess-1757705202/manifest/2025-09-12/19.jsonl.enc"
master_key_file = "secure_store/master.key"

# --- LOAD MASTER KEY ---
def load_master_key(path):
    txt = Path(path).read_text().strip()
    try:
        return base64.b64decode(txt)
    except Exception:
        return Path(path).read_bytes()

master_key = load_master_key(master_key_file)

# --- DERIVE KEY (context = parent dir of enc_file) ---
enc_path = Path(enc_file)
context = str(enc_path.parent)
hkdf = HKDF(
    algorithm=hashes.SHA256(),
    length=32,
    salt=None,
    info=context.encode(),
)
key = hkdf.derive(master_key)

# --- LOAD ENCRYPTED PAYLOAD ---
payload = json.loads(enc_path.read_text())
nonce = base64.b64decode(payload["nonce"])
ct = base64.b64decode(payload["ct"])

# --- DECRYPT ---
aesgcm = AESGCM(key)
plaintext = aesgcm.decrypt(nonce, ct, None)

# --- PRINT OR SAVE ---
print("Decrypted bytes:", plaintext)
print("As UTF-8:", plaintext.decode("utf-8"))