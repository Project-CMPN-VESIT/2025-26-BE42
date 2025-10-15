# decrypt.py
import os
import shutil
import json
from centralized_secure_store import SecureStore

def walk_and_decrypt(store: SecureStore, root: str, out_root: str):
    """
    Recursively walk through the SecureStore directory tree.
      - Non-encrypted files (.enc missing) are copied as-is.
      - Encrypted files (.enc) are decrypted using CentralizedSecureStore
        and written to the output directory with `.enc` stripped.
    """

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            enc_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(enc_path, root)
            out_path = os.path.join(out_root, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Non-encrypted file → direct copy
            if not fn.endswith(".enc"):
                shutil.copy2(enc_path, out_path)
                print(f"[COPY] {enc_path}  →  {out_path}")
                continue

            # Encrypted file → decrypt
            try:
                # CentralizedSecureStore expects a file:// URI
                decrypted_bytes = store.decrypt_read(f"file://{enc_path}")
                out_file = out_path[:-4]  # strip '.enc'
                with open(out_file, "wb") as f:
                    f.write(decrypted_bytes)
                print(f"[DECRYPT] {enc_path}  →  {out_file}")
            except Exception as e:
                print(f"[ERROR] {enc_path}  →  DECRYPT_FAILED ({type(e).__name__}: {e})")

def main():
    in_root = "./secure_store"
    out_root = "./decrypted_output"

    if not os.path.exists(in_root):
        print(f"[ERROR] Input directory not found: {in_root}")
        return

    os.makedirs(out_root, exist_ok=True)

    # Initialize centralized key store
    # Adjust the key path if your centralized_secure_store.py expects a config file or key path
    store = SecureStore(root_path=in_root)

    print(f"=== Starting recursive decryption from {in_root} ===\n")
    walk_and_decrypt(store, in_root, out_root)
    print("\n=== Decryption complete ===")
    print(f"Decrypted output available at: {os.path.abspath(out_root)}")

if __name__ == "__main__":
    main()
