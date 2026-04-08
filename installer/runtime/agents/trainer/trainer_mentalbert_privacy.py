"""
trainer_agent/trainer_mentalbert_privacy.py

Multimodal trainer agent supporting:
 - autonomous (inference + explainability)
 - supervised (physician-corrected fine-tune -> delta)
 - rl (online REINFORCE-style updates based on clinician reward)

Safety policy (applied automatically):
 - per-parameter absolute delta clamp (max_param_change)
 - global delta L2 norm scaling (max_global_delta_norm)
 - gradient clipping during optimization
 - conservative default learning rates and small batch sizes for RL

Usage:
  python trainer_mentalbert_privacy.py --mode autonomous --input ./session.parquet
  python trainer_mentalbert_privacy.py --mode supervised --input ./session.parquet --epochs 2
  python trainer_mentalbert_privacy.py --mode rl --input ./session.parquet --epochs 1
"""

import os
import sys
import json
import argparse
import math
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW

import pandas as pd
import pyarrow.parquet as pq

from core.centralized_secure_store import SecureStore
from core.centralised_receipts import CentralReceiptManager

from installer.security.integrity import integrity_guard
integrity_guard()

# ---------- Config / Defaults ----------
MENTALBERT_PRETRAIN = "mental/mental-bert-base-uncased"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_SAVE_DIR = Path.home() / ".federated" / "data" / "secure_store"
LOCAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Safety hyperparameters (tunable)
DEFAULT_MAX_PARAM_CHANGE = 1e-3        # per-parameter absolute clamp on delta
DEFAULT_MAX_GLOBAL_DELTA_NORM = 1.0   # max L2 norm of delta state (after per-param clamp will be scaled down to this)
RL_PHQ_RANGE = 30.0                   # normalization range for PHQ when computing reward


# ---------- Dataset ----------
class MultiModalDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_len: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def _extract_audio_vec(self, r):
        feats = r.get("features") or {}
        audio = feats.get("audio")
        if not isinstance(audio, dict):
            return None

        # 1) Prefer wav2vec2 embeddings if present
        if isinstance(audio.get("wav2vec2"), list):
            return torch.tensor(audio["wav2vec2"], dtype=torch.float32)

        # 2) Fallback: use all numeric audio features
        numeric_vals = []
        for k, v in audio.items():
            if isinstance(v, (int, float)):
                numeric_vals.append(float(v))

        if numeric_vals:
            return torch.tensor(numeric_vals, dtype=torch.float32)

        return None

    def _extract_video_vec(self, r):
        feats = r.get("features") or {}
        if isinstance(feats, dict) and "video" in feats and isinstance(feats["video"], dict):
            v = feats["video"]
            if v.get("densenet") and isinstance(v["densenet"], (list, tuple)):
                return torch.tensor(v["densenet"], dtype=torch.float32)
            if v.get("densenet_csv") and isinstance(v["densenet_csv"], str):
                try:
                    arr = [float(x) for x in v["densenet_csv"].split(",") if x.strip() != ""]
                    return torch.tensor(arr, dtype=torch.float32)
                except:
                    return None
        # fallback: check top-level neuron_* keys
        neuron_keys = sorted([k for k in r.keys() if str(k).startswith("neuron_")])
        if neuron_keys:
            arr = [float(r[k]) for k in neuron_keys]
            return torch.tensor(arr, dtype=torch.float32)
        return None

    def __getitem__(self, idx):
        r = self.records[idx]
        def _safe_text(r) -> str:
            """Return transcript text, defaulting to empty string."""
            return (r.get("transcript") or r.get("text") or "").strip()

        text = _safe_text(r)
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        audio_vec = self._extract_audio_vec(r)
        video_vec = self._extract_video_vec(r)

        phq = r.get("phq_score") or r.get("phq") or r.get("target_phq") or r.get("label_phq")
        try:
            phq_val = float(phq) if phq is not None else 0.0
        except:
            phq_val = 0.0
        label = 1 if phq_val >= 10.0 else 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_vec": audio_vec,
            "video_vec": video_vec,
            "phq": torch.tensor(phq_val, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------- Model components ----------
class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, hidden: Optional[int] = None, dropout: float = 0.2):
        super().__init__()
        hid = hidden or max(32, in_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden, num_classes)
        # For RL policy, we output mu and log_sigma for PHQ (continuous action)
        self.phq_mu = nn.Linear(hidden, 1)
        self.phq_logsigma = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.drop(self.act(self.fc1(x)))
        logits = self.classifier(h)
        mu = self.phq_mu(h).squeeze(-1)
        log_sigma = self.phq_logsigma(h).squeeze(-1)
        return logits, mu, log_sigma


class MultiModalModel(nn.Module):
    def __init__(self, bert_name: str, audio_dim: Optional[int], vision_dim: Optional[int], device: str = DEFAULT_DEVICE):
        super().__init__()
        self.device = device
        self.bert = AutoModel.from_pretrained(bert_name)
        bert_hidden = self.bert.config.hidden_size

        self.has_audio = audio_dim is not None and audio_dim > 0
        self.has_vision = vision_dim is not None and vision_dim > 0

        self.audio_encoder = SmallMLP(audio_dim, out_dim=128) if self.has_audio else None
        self.vision_encoder = SmallMLP(vision_dim, out_dim=128) if self.has_vision else None

        fusion_input_dim = bert_hidden + (128 if self.has_audio else 0) + (128 if self.has_vision else 0)
        self.fusion = FusionHead(fusion_input_dim)

    def forward(self, input_ids, attention_mask, audio_vec=None, vision_vec=None, rl_mode=False, sample_action=False):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = bert_out.last_hidden_state[:, 0, :]

        audio_enc = None
        vision_enc = None
        if self.has_audio:
            if audio_vec is not None:
                audio_enc = self.audio_encoder(audio_vec)
            else:
                audio_enc = torch.zeros((pooled.size(0), 128), device=pooled.device)
        if self.has_vision:
            if vision_vec is not None:
                vision_enc = self.vision_encoder(vision_vec)
            else:
                vision_enc = torch.zeros((pooled.size(0), 128), device=pooled.device)

        parts = [pooled]
        if audio_enc is not None:
            parts.append(audio_enc)
        if vision_enc is not None:
            parts.append(vision_enc)
        fused = torch.cat(parts, dim=1)

        logits, mu, log_sigma = self.fusion(fused)
        # reg prediction for supervised MSE is mu (deterministic)
        reg_pred = mu

        # RL: if sampling, draw action from N(mu, sigma)
        if rl_mode and sample_action:
            sigma = torch.exp(log_sigma).clamp(min=1e-4)
            noise = torch.randn_like(mu)
            action = mu + sigma * noise
            # compute log_prob manually
            var = sigma * sigma
            log_prob = -0.5 * (((action - mu) ** 2) / var + 2 * log_sigma + math.log(2 * math.pi))
            return logits, reg_pred, (action, log_prob)
        else:
            return logits, reg_pred, (mu, log_sigma)


# ---------- Utilities ----------
def read_parquet_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # -------- Load records --------
    if p.suffix == ".jsonl":
        rows = []
        _CANONICAL_STORE = Path.home() / ".federated" / "data" / "secure_store"
 
        store = SecureStore(
            agent="lda",
            root=_CANONICAL_STORE,
        )

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)

                enc_uri = m["uri"]
                row_idx = m["row"]

                # decrypt parquet
                parquet_bytes = store.decrypt_read(enc_uri)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tf:
                    tf.write(parquet_bytes)
                    temp_path = tf.name

                # 🔥 IMPORTANT: file is CLOSED here

                table = pq.read_table(temp_path)

                # cleanup
                os.remove(temp_path)
                df = table.to_pandas()
                rows.append(df.iloc[row_idx].to_dict())

        records = rows
    elif p.suffix == ".parquet":
        table = pq.read_table(str(p))
        df = table.to_pandas()
        records = df.to_dict(orient="records")
    elif p.suffix in (".json", ".jsonl"):
        records = []
        with open(p, "r", encoding="utf-8") as f:
            if p.suffix == ".json":
                records = json.load(f)
            else:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
    else:
        df = pd.read_csv(str(p))
        records = df.to_dict(orient="records")

    # -------- Parse embedded JSON fields --------
    for r in records:
        for key in ("features", "derived"):
            if key in r and isinstance(r[key], str):
                try:
                    r[key] = json.loads(r[key])
                except Exception:
                    pass

    # -------- FILTER BAD TRANSCRIPTS (FIXED) --------
    def _filter_records(records):
        """
        Filter records before training.
        - "failed": ASR crashed entirely (infrastructure problem) → drop
        - "empty": no speech detected in segment (normal) → keep with empty text
        - "ok": transcript present → keep
        - missing key: keep (older data without status field)
        """
        filtered = []
        dropped = 0
        for r in records:
            derived = r.get("derived") or {}
            # derived may be a JSON string (from parquet)
            if isinstance(derived, str):
                try:
                    import json
                    derived = json.loads(derived)
                except Exception:
                    derived = {}
            status = derived.get("transcript_status", "ok")
            if status == "failed":
                dropped += 1
                continue
            filtered.append(r)
    
        if dropped:
            import logging
            logging.getLogger(__name__).warning(
                "Dropped %d records with transcript_status=failed", dropped
            )
    
        if not filtered:
            raise RuntimeError(
                "All records were dropped (transcript_status=failed for all). "
                "Check your ASR setup."
            )
        return filtered

def collate_batch(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)

    # pad audio vectors
    audio_list = [b["audio_vec"] for b in batch]
    if any(a is not None for a in audio_list):
        max_a = max([a.size(0) if a is not None else 0 for a in audio_list])
        padded_as = []
        for a in audio_list:
            if a is None:
                padded_as.append(torch.zeros(max_a))
            elif a.size(0) < max_a:
                padded_as.append(torch.cat([a, torch.zeros(max_a - a.size(0))], dim=0))
            else:
                padded_as.append(a)
        audio_batch = torch.stack(padded_as, dim=0)
    else:
        audio_batch = None

    # pad video vectors
    video_list = [b["video_vec"] for b in batch]
    if any(v is not None for v in video_list):
        max_v = max([v.size(0) if v is not None else 0 for v in video_list])
        padded_vs = []
        for v in video_list:
            if v is None:
                padded_vs.append(torch.zeros(max_v))
            elif v.size(0) < max_v:
                padded_vs.append(torch.cat([v, torch.zeros(max_v - v.size(0))], dim=0))
            else:
                padded_vs.append(v)
        video_batch = torch.stack(padded_vs, dim=0)
    else:
        video_batch = None

    phq = torch.stack([b["phq"] for b in batch], dim=0)
    label = torch.stack([b["label"] for b in batch], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_vec": audio_batch,
        "video_vec": video_batch,
        "phq": phq,
        "label": label,
    }


# ---------- Training / Inference ----------
def run_inference(model: MultiModalModel, dataloader: DataLoader, device: str = DEFAULT_DEVICE):
    model.eval()
    results = []
    with torch.no_grad():
        for b in dataloader:
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            logits, reg_pred, _ = model(inputs["input_ids"], inputs["attention_mask"],
                                       audio_vec=inputs.get("audio_vec"), vision_vec=inputs.get("video_vec"),
                                       rl_mode=False, sample_action=False)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds_class = probs.argmax(axis=1)
            preds_phq = reg_pred.cpu().numpy()
            for i in range(preds_phq.shape[0]):
                results.append({
                    "pred_class": int(preds_class[i]),
                    "pred_class_probs": probs[i].tolist(),
                    "pred_phq": float(preds_phq[i]),
                })
    return results


def fine_tune_supervised(model: MultiModalModel, dataset: MultiModalDataset, epochs: int = 1, batch_size: int = 8, lr: float = 2e-5, device: str = DEFAULT_DEVICE):
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    optimizer = AdamW(model.parameters(), lr=lr)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            optimizer.zero_grad()
            logits, reg_pred, _ = model(batch["input_ids"], batch["attention_mask"],
                                        audio_vec=batch.get("audio_vec"), vision_vec=batch.get("video_vec"))
            loss_cls = cls_loss_fn(logits, batch["label"])
            loss_reg = reg_loss_fn(reg_pred, batch["phq"])
            loss = loss_cls + 0.5 * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        if steps:
            print(f"[supervised] epoch {epoch+1}/{epochs} avg_loss={total_loss/steps:.4f}")
    return model

# ---------- Unified supervised training + evaluation + explainability ----------
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error

def train_model(dataset: MultiModalDataset, model: MultiModalModel,
                output_dir: str = "./trainer_outputs",
                epochs: int = 5, batch_size: int = 8, lr: float = 2e-5,
                device: str = DEFAULT_DEVICE):
    """
    Complete supervised fine-tuning on labeled PHQ data + evaluation + explainability.
    Produces model weights, metrics.json, and explain.txt.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    optimizer = AdamW(model.parameters(), lr=lr)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for b in loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            optimizer.zero_grad()
            logits, reg_pred, _ = model(b["input_ids"], b["attention_mask"],
                                        audio_vec=b.get("audio_vec"), vision_vec=b.get("video_vec"))
            loss_cls = cls_loss_fn(logits, b["label"])
            loss_reg = reg_loss_fn(reg_pred, b["phq"])
            loss = loss_cls + 0.5 * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"[train_model] epoch {epoch+1}/{epochs} avg_loss={total_loss/len(loader):.4f}")

    # ---------- Evaluation ----------
    model.eval()
    y_true_cls, y_pred_cls = [], []
    y_true_phq, y_pred_phq = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            logits, reg_pred, _ = model(b["input_ids"], b["attention_mask"],
                                        audio_vec=b.get("audio_vec"), vision_vec=b.get("video_vec"))
            probs = torch.softmax(logits, dim=1)
            preds_cls = probs.argmax(dim=1)
            y_true_cls.extend(b["label"].cpu().tolist())
            y_pred_cls.extend(preds_cls.cpu().tolist())
            y_true_phq.extend(b["phq"].cpu().tolist())
            y_pred_phq.extend(reg_pred.cpu().tolist())

    acc = accuracy_score(y_true_cls, y_pred_cls) if y_true_cls else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average="binary") if y_true_cls else (0.0, 0.0, 0.0, None)
    mae = mean_absolute_error(y_true_phq, y_pred_phq) if y_true_phq else 0.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mae": float(mae),
    }

    # ---------- Explainability ----------
    explain_path = os.path.join(output_dir, "explain.txt")
    try:
        with open(explain_path, "w") as f:
            f.write("=== Explainability Report ===\n")
            f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n\n")
            try:
                first_batch = next(iter(loader))
                importance = modality_ablation_importance(model, first_batch, device=device)
                f.write("Modality contributions (approx):\n")
                for k, v in importance["raw"].items():
                    f.write(f"  {k}: {v:.4f}\n")
            except Exception as e:
                f.write(f"[WARN] Explainability failed: {e}\n")
    except Exception as e:
        print(f"[WARN] Could not write explain.txt: {e}")

    # ---------- Save everything ----------
    model_path = os.path.join(output_dir, "mentalbert_privacy_subset.pt")
    torch.save(model.state_dict(), model_path)

    # ---- write metrics.json for downstream consumers ----
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    try:
        with open(metrics_json_path, "w") as mf:
            json.dump(metrics, mf, indent=2)
        print(f"[train_model] metrics saved → {metrics_json_path}")
    except Exception as e:
        print(f"[WARN] Could not write metrics.json: {e}")

    # Also write a receipt-like file (existing behavior)
    with open(os.path.join(output_dir, "metrics_report.json"), "w") as f:
        json.dump({"model_path": model_path, "metrics": metrics}, f, indent=2)

    print(f"[train_model] model saved → {model_path}")
    return {
        "model_path": model_path,
        "metrics_path": metrics_json_path,
        "explain_path": explain_path,
        "metrics": metrics
    }

# ---------- RL Update (REINFORCE) ----------
class MovingBaseline:
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.value = 0.0
        self.inited = False

    def update(self, r: float):
        if not self.inited:
            self.value = r
            self.inited = True
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * r
        return self.value

def rl_update_reinforce(model: MultiModalModel, dataset: MultiModalDataset, epochs: int = 1, batch_size: int = 4, lr: float = 1e-5, device: str = DEFAULT_DEVICE, supervised_lambda: float = 0.0):
    """
    Simple REINFORCE per-sample using clinician-corrected PHQ as reward signal.
    supervised_lambda: mixing coefficient to add supervised MSE loss (helps stabilize).
    """
    model.to(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_batch)  # sample-level for RL
    optimizer = AdamW(model.parameters(), lr=lr)
    baseline = MovingBaseline(momentum=0.9)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            # each batch will be size 1 (sample-level)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # sample an action
            logits, reg_pred, (action, log_prob) = model(batch["input_ids"], batch["attention_mask"],
                                                        audio_vec=batch.get("audio_vec"), vision_vec=batch.get("video_vec"),
                                                        rl_mode=True, sample_action=True)
            # clinician-corrected PHQ is in batch["phq"]
            target_phq = batch["phq"].squeeze(0).item()
            # compute reward in [0,1]
            error = abs(action.squeeze(0).item() - target_phq)
            norm_error = min(error / RL_PHQ_RANGE, 1.0)
            reward = 1.0 - norm_error

            # baseline subtraction
            b = baseline.update(reward)
            adv = reward - b

            # REINFORCE loss (negative reward-weighted logprob)
            logp = log_prob.squeeze(0)
            rl_loss = -adv * logp.mean()

            # optional supervised MSE loss
            mse_loss = nn.functional.mse_loss(action.squeeze(0), batch["phq"])
            loss = rl_loss + supervised_lambda * mse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
        if steps:
            print(f"[rl] epoch {epoch+1}/{epochs} avg_loss={total_loss/steps:.6f}")
    return model


# ---------- Delta computation & safety ----------
def compute_state_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    delta = {}
    for k in after:
        if k in before and before[k].shape == after[k].shape:
            delta[k] = (after[k].detach().cpu() - before[k].detach().cpu()).clone()
        else:
            delta[k] = after[k].detach().cpu().clone()
    return delta

def apply_safety_to_delta(delta: Dict[str, torch.Tensor], max_param_change: float = DEFAULT_MAX_PARAM_CHANGE, max_global_norm: float = DEFAULT_MAX_GLOBAL_DELTA_NORM) -> Dict[str, torch.Tensor]:
    # per-parameter clamp
    for k in list(delta.keys()):
        t = delta[k]
        delta[k] = t.clamp(min=-max_param_change, max=max_param_change)

    # compute global norm
    total_sq = 0.0
    for k in delta:
        total_sq += (delta[k].float().norm() ** 2).item()
    global_norm = math.sqrt(total_sq)

    if global_norm > max_global_norm:
        scale = max_global_norm / (global_norm + 1e-12)
        for k in delta:
            delta[k] = (delta[k].float() * scale).clone()
    return delta

def save_encrypted_delta(delta_state: Dict[str, torch.Tensor], store: SecureStore, session_id: str, rm: CentralReceiptManager) -> Tuple[str, str]:
    import io, torch as _torch
    buf = io.BytesIO()
    _torch.save(delta_state, buf)
    payload = buf.getvalue()
    rel = f"{session_id}/local_updates/{os.urandom(8).hex()}.pt.enc"
    uri = store.encrypt_write(f"file://{store.root / rel}", payload)
    receipt = rm.create_receipt(
        agent="trainer-agent",
        operation="local_update",
        params={"size_bytes": len(payload)},
        outputs=[uri],
        session_id=session_id,
    )
    rrel = f"{session_id}/receipts/trainer_update_{os.urandom(6).hex()}.json.enc"
    ruri = store.encrypt_write(f"file://{store.root / rrel}", json.dumps(receipt).encode())
    return uri, ruri


# ---------- Explainability (modality ablation) ----------
def modality_ablation_importance(model: MultiModalModel, sample_batch: Dict[str, torch.Tensor], device: str = DEFAULT_DEVICE):
    model.eval()
    with torch.no_grad():
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample_batch.items()}
        logits, reg, _ = model(inputs["input_ids"], inputs["attention_mask"], audio_vec=inputs.get("audio_vec"), vision_vec=inputs.get("video_vec"))
        base_pos = torch.softmax(logits, dim=1)[:, 1].mean().item()
        base_phq = reg.mean().item()
        results = {}
        # audio ablation
        if inputs.get("audio_vec") is not None:
            zero_a = torch.zeros_like(inputs["audio_vec"])
            l_a, r_a, _ = model(inputs["input_ids"], inputs["attention_mask"], audio_vec=zero_a, vision_vec=inputs.get("video_vec"))
            p_a = torch.softmax(l_a, dim=1)[:, 1].mean().item()
            r_a_val = r_a.mean().item()
            results["audio_posdelta"] = abs(base_pos - p_a)
            results["audio_phqdelta"] = abs(base_phq - r_a_val)
        else:
            results["audio_posdelta"] = 0.0
            results["audio_phqdelta"] = 0.0
        # vision ablation
        if inputs.get("video_vec") is not None:
            zero_v = torch.zeros_like(inputs["video_vec"])
            l_v, r_v, _ = model(inputs["input_ids"], inputs["attention_mask"], audio_vec=inputs.get("audio_vec"), vision_vec=zero_v)
            p_v = torch.softmax(l_v, dim=1)[:, 1].mean().item()
            r_v_val = r_v.mean().item()
            results["vision_posdelta"] = abs(base_pos - p_v)
            results["vision_phqdelta"] = abs(base_phq - r_v_val)
        else:
            results["vision_posdelta"] = 0.0
            results["vision_phqdelta"] = 0.0
        # text ablation via empty string
        try:
            tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
            empty = tokenizer([""] * inputs["input_ids"].shape[0], padding=True, truncation=True, return_tensors="pt")
            empty_ids = empty["input_ids"].to(device)
            empty_mask = empty["attention_mask"].to(device)
            l_t, r_t, _ = model(empty_ids, empty_mask, audio_vec=inputs.get("audio_vec"), vision_vec=inputs.get("video_vec"))
            p_t = torch.softmax(l_t, dim=1)[:, 1].mean().item()
            r_t_val = r_t.mean().item()
            results["text_posdelta"] = abs(base_pos - p_t)
            results["text_phqdelta"] = abs(base_phq - r_t_val)
        except Exception:
            results["text_posdelta"] = 0.0
            results["text_phqdelta"] = 0.0

        agg = {
            "audio_score": (results["audio_posdelta"] + results["audio_phqdelta"]) / 2.0,
            "vision_score": (results["vision_posdelta"] + results["vision_phqdelta"]) / 2.0,
            "text_score": (results["text_posdelta"] + results["text_phqdelta"]) / 2.0,
            "raw": results
        }
        return agg


# ---------- Physician CLI for supervised correction ----------
def physician_feedback_cli(preds: List[Dict[str, Any]], texts: List[str]) -> List[float]:
    corrected = []
    print("\n-- Physician feedback loop --")
    for i, (p, t) in enumerate(zip(preds, texts)):
        snippet = (t or "")[:260].replace("\n", " ")
        print(f"\nSample {i+1}: {snippet} ...")
        print(f"Model PHQ: {p['pred_phq']:.2f}  (class prob positive: {p['pred_class_probs'][1]:.3f})")
        val = input("Enter corrected PHQ (or Enter to keep): ").strip()
        if val:
            try:
                corrected.append(float(val))
            except:
                print("invalid -> keeping model value")
                corrected.append(float(p["pred_phq"]))
        else:
            corrected.append(float(p["pred_phq"]))
    return corrected


# ---------- Orchestrator ----------
def orchestrate(
    input_path: str,
    session_id: str,
    mode: str = "autonomous",
    device: str = None,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 2e-5,
    rl_supervised_lambda: float = 0.0,
    max_samples=None,
    safety_params=None,
    global_model_path: str = None,   # ← Phase 10: now an explicit parameter
    **kwargs
):
    """
    Orchestrate training/inference.
 
    global_model_path: optional path to a global model state_dict (.pt file)
                       received from the server.  When provided the model is
                       initialised from these weights before local fine-tuning,
                       implementing the federated averaging warm-start.
    """
    import os, json, tempfile, torch
    from pathlib import Path
    from torch.utils.data import DataLoader
 
    # resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # ── inline imports so the rest of the file remains unchanged ─────────────
    from transformers import AutoTokenizer
 
    # These are defined in the same file — import via module self-reference
    import sys
    _mod = sys.modules[__name__]
 
    read_parquet_records    = _mod.read_parquet_records
    MultiModalDataset       = _mod.MultiModalDataset
    MultiModalModel         = _mod.MultiModalModel
    collate_batch           = _mod.collate_batch
    run_inference           = _mod.run_inference
    train_model             = _mod.train_model
    rl_update_reinforce     = _mod.rl_update_reinforce
    compute_state_delta     = _mod.compute_state_delta
    apply_safety_to_delta   = _mod.apply_safety_to_delta
    save_encrypted_delta    = _mod.save_encrypted_delta
    modality_ablation_importance = _mod.modality_ablation_importance
    physician_feedback_cli  = _mod.physician_feedback_cli
    SecureStore             = _mod.SecureStore
    CentralReceiptManager   = _mod.CentralReceiptManager
    MENTALBERT_PRETRAIN     = _mod.MENTALBERT_PRETRAIN
    LOCAL_SAVE_DIR          = _mod.LOCAL_SAVE_DIR
    DEFAULT_MAX_PARAM_CHANGE     = _mod.DEFAULT_MAX_PARAM_CHANGE
    DEFAULT_MAX_GLOBAL_DELTA_NORM = _mod.DEFAULT_MAX_GLOBAL_DELTA_NORM
 
    # ── load records ──────────────────────────────────────────────────────────
    if input_path.startswith("file://") and input_path.endswith(".enc"):
        _CANONICAL_STORE = Path.home() / ".federated" / "data" / "secure_store"
        store_lda = SecureStore(agent="lda", root=_CANONICAL_STORE)
        manifest_bytes = store_lda.decrypt_read(input_path)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            tf.write(manifest_bytes)
            local_manifest_path = tf.name
        records = read_parquet_records(local_manifest_path)
    else:
        records = read_parquet_records(input_path)
 
    if max_samples:
        records = records[:max_samples]
    print(f"[info] loaded {len(records)} records")
 
    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
    ds        = MultiModalDataset(records, tokenizer)
 
    # infer modality dims
    audio_dim  = None
    vision_dim = None
    for r in records:
        f     = r.get("features") or {}
        audio = f.get("audio")
        if isinstance(audio, dict):
            if isinstance(audio.get("wav2vec2"), list):
                audio_dim = len(audio["wav2vec2"])
                break
            numeric = [v for v in audio.values() if isinstance(v, (int, float))]
            if numeric:
                audio_dim = len(numeric)
                break
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "video" in f and isinstance(f["video"], dict):
            v = f["video"]
            if v.get("densenet") and isinstance(v["densenet"], (list, tuple)):
                vision_dim = len(v["densenet"])
                break
        if any(k.startswith("neuron_") for k in r.keys()):
            vision_dim = len(sorted(k for k in r if k.startswith("neuron_")))
            break
 
    print(f"[info] inferred audio_dim={audio_dim}, vision_dim={vision_dim}")
 
    model = MultiModalModel(
        MENTALBERT_PRETRAIN,
        audio_dim=audio_dim,
        vision_dim=vision_dim,
        device=device
    )
 
    # ── Phase 10: warm-start from global model ────────────────────────────────
    if global_model_path and Path(global_model_path).exists():
        try:
            global_state = torch.load(global_model_path, map_location=device)
            # Support both raw state_dict and wrapped {"state_dict": ...} formats
            if isinstance(global_state, dict) and "state_dict" in global_state:
                global_state = global_state["state_dict"]
            missing, unexpected = model.load_state_dict(global_state, strict=False)
            print(
                f"[FL] Warm-started from global model: "
                f"{len(missing)} missing keys, {len(unexpected)} unexpected keys"
            )
        except Exception as e:
            print(f"[FL] Warning: could not load global model ({e}) — starting from scratch")
    else:
        if global_model_path:
            print(f"[FL] global_model_path provided but file not found: {global_model_path}")
        print("[FL] No global model — using random initialisation")
    # ─────────────────────────────────────────────────────────────────────────
 
    model.to(device)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
 
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    preds  = run_inference(model, loader, device=device)
    for r, p in zip(records, preds):
        r["_pred_phq"]         = p["pred_phq"]
        r["_pred_class_probs"] = p["pred_class_probs"]
 
    store = SecureStore(agent="trainer", root=LOCAL_SAVE_DIR)
    rm    = CentralReceiptManager(agent="trainer-agent")
 
    if mode == "autonomous":
        sample = next(iter(loader), None)
        explanations = modality_ablation_importance(model, sample, device=device) if sample else {}
        payload   = json.dumps({"preds": preds, "explainability": explanations}, default=float).encode()
        out_rel   = f"{session_id}/inference/results_{os.urandom(6).hex()}.json.enc"
        out_uri   = store.encrypt_write(f"file://{store.root / out_rel}", payload)
        receipt   = rm.create_receipt(
            agent="trainer-agent", operation="inference",
            params={"count": len(preds)}, outputs=[out_uri], session_id=session_id,
        )
        rrel = f"{session_id}/receipts/inference_{os.urandom(6).hex()}.json.enc"
        ruri = store.encrypt_write(f"file://{store.root / rrel}", json.dumps(receipt).encode())
        print(f"[done] inference -> {out_uri}")
        return {"inference_uri": out_uri, "receipt_uri": ruri, "explainability": explanations}
 
    elif mode == "supervised":
        texts         = [r.get("transcript") or r.get("text") or "" for r in records]
        corrected_phq = physician_feedback_cli(preds, texts)
        for r, cp in zip(records, corrected_phq):
            r["phq_score"] = float(cp)
        ds_sup = MultiModalDataset(records, tokenizer)
        result = train_model(ds_sup, model, output_dir=str(LOCAL_SAVE_DIR),
                             epochs=epochs, batch_size=batch_size, lr=lr, device=device)
        model.load_state_dict(torch.load(result["model_path"], map_location=device))
        after  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        delta  = compute_state_delta(base_state, after)
        sparams = safety_params or {}
        delta_safe = apply_safety_to_delta(
            delta,
            max_param_change=sparams.get("max_param_change", DEFAULT_MAX_PARAM_CHANGE),
            max_global_norm=sparams.get("max_global_norm", DEFAULT_MAX_GLOBAL_DELTA_NORM),
        )
        update_uri, receipt_uri = save_encrypted_delta(delta_safe, store, session_id, rm)
        print(f"[done] supervised training -> delta saved {update_uri}")
        return {"local_update_uri": update_uri, "update_receipt_uri": receipt_uri}
 
    elif mode == "rl":
        texts    = [r.get("transcript") or r.get("text") or "" for r in records]
        need_cli = not any(r.get("phq_score") is not None for r in records)
        if need_cli:
            corrected_phq = physician_feedback_cli(preds, texts)
            for r, cp in zip(records, corrected_phq):
                r["phq_score"] = float(cp)
        else:
            for r in records:
                r["phq_score"] = float(r.get("phq_score") or r.get("phq") or 0.0)
        ds_rl = MultiModalDataset(records, tokenizer)
        model = rl_update_reinforce(model, ds_rl, epochs=epochs, batch_size=1,
                                    lr=lr, device=device,
                                    supervised_lambda=rl_supervised_lambda)
        after  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        delta  = compute_state_delta(base_state, after)
        sparams = safety_params or {}
        delta_safe = apply_safety_to_delta(
            delta,
            max_param_change=sparams.get("max_param_change", DEFAULT_MAX_PARAM_CHANGE),
            max_global_norm=sparams.get("max_global_norm", DEFAULT_MAX_GLOBAL_DELTA_NORM),
        )
        update_uri, receipt_uri = save_encrypted_delta(delta_safe, store, session_id, rm)
        print(f"[done] RL training -> delta saved {update_uri}")
        return {"local_update_uri": update_uri, "update_receipt_uri": receipt_uri}
 
    else:
        raise ValueError("mode must be 'autonomous'|'supervised'|'rl'")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Trainer Agent (MentalBERT multimodal) with supervised + RL modes")
    parser.add_argument("--mode", choices=["autonomous", "supervised", "rl"], required=True)
    parser.add_argument("--input", required=True, help="parquet/json/csv path from LDA")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--rl-supervised-lambda", type=float, default=0.0, help="mix supervised MSE into RL updates")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-param-change", type=float, default=DEFAULT_MAX_PARAM_CHANGE)
    parser.add_argument("--max-global-delta-norm", type=float, default=DEFAULT_MAX_GLOBAL_DELTA_NORM)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()

    safety = {"max_param_change": args.max_param_change, "max_global_norm": args.max_global_delta_norm}
    res = orchestrate(
        input_path=args.input,
        session_id=args.session_id,
        mode=args.mode,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rl_supervised_lambda=args.rl_supervised_lambda,
        max_samples=args.max_samples,
        safety_params=safety
    )
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
