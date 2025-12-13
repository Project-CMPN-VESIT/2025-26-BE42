import torch
import numpy as np
import io
from centralized_secure_store import SecureStore
from server.aggregator_agent.aggregator import AggregatorAgent

store = SecureStore("./secure_store")

# Create encrypted update files
updates = []
for i, vec in enumerate([
    np.array([1,2,3], float),
    np.array([1.1,2.1,2.9], float),
    np.array([10,-9,100], float)
]):
    # 1. Convert to torch tensor
    t = torch.tensor(vec, dtype=torch.float32)

    # 2. Serialize to bytes (unencrypted)
    buf = io.BytesIO()
    torch.save(t, buf)
    buf.seek(0)
    raw_bytes = buf.getvalue()

    # 3. Encrypt using SecureStore AES-GCM
    fname = f"secure_store/tmp_update_{i}.pt.enc"
    uri = "file://" + fname
    store.encrypt_write(uri, raw_bytes)

    # 4. Append aggregator packet
    updates.append({
        "client_id": f"c{i}",
        "enc_uri": uri,
        "scheme": "AES-GCM-SecureStore",   # IMPORTANT
        "nonce": None,
        "receipt": {},
        "metadata": {}
    })

# Run aggregator
agg = AggregatorAgent(mode="trimmed_mean", trim_ratio=0.33)
result = agg.aggregate_updates(updates)

print("Aggregated:", result)
