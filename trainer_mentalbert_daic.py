#!/usr/bin/env python3
# trainer_mentalbert_daic.py
"""
Standalone trainer / inference / explainability script for MentalBERT on DAIC-WOZ-like data.

Features:
 - Loads parquet/json/csv of session records (transcript, features.{audio,video}, phq fields)
 - Multimodal model: MentalBERT (from transformers) + small MLPs for audio/video vectors + fusion head
 - Supervised fine-tune: classification (binary PHQ>threshold) + regression (PHQ score)
 - Evaluation metrics: accuracy/precision/recall/f1 for classification; MAE for regression
 - Explainability:
    - modality ablation importance (text/audio/video)
    - linear-probe explanation on final embeddings (top feature contributions)
    - per-sample explanation logs (CSV and human-readable .txt)
 - Saves: model state_dict, delta (after safety clamps), receipts (json)
Usage example:
    python trainer_mentalbert_daic.py --mode supervised --input ./data/session.parquet \
        --device cuda --epochs 4 --batch-size 8 --out-path ./trainer_outputs/mentalbert_privacy_subset.pt
"""

import os
import json
import time
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression

import pyarrow.parquet as pq

# Simple SecureStore fallback (non-encrypted)
class SecureStoreFallback:
    def __init__(self, root="./trainer_outputs/secure_store", agent="trainer-agent"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.agent = agent

    def encrypt_write(self, uri: str, payload: bytes) -> str:
        # Accepts file:// or direct path
        if uri.startswith("file://"):
            p = Path(uri[len("file://"):])
        else:
            p = Path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
        return f"file://{p}"

    def decrypt_read(self, uri: str) -> bytes:
        if uri.startswith("file://"):
            p = Path(uri[len("file://"):])
        else:
            p = Path(uri)
        return p.read_bytes()

# -------------------------
# Dataset & utilities
# -------------------------
def read_parquet_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    if p.suffix == ".parquet":
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

    # try to decode JSON strings in features/derived
    for r in records:
        for key in ("features", "derived"):
            if key in r and isinstance(r[key], str):
                try:
                    r[key] = json.loads(r[key])
                except Exception:
                    pass
    return records


class MultiModalDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_len: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def _extract_audio_vec(self, r):
        feats = r.get("features") or {}
        if isinstance(feats, dict) and "audio" in feats:
            a = feats["audio"]
            if isinstance(a, dict) and a.get("wav2vec2"):
                return torch.tensor(a["wav2vec2"], dtype=torch.float32)
            if isinstance(a, dict) and a.get("egemaps"):
                vals = [float(a["egemaps"][k]) for k in sorted(a["egemaps"].keys())]
                return torch.tensor(vals, dtype=torch.float32)
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
        # fallback check top-level neuron_* keys
        neuron_keys = sorted([k for k in r.keys() if str(k).startswith("neuron_")])
        if neuron_keys:
            arr = [float(r[k]) for k in neuron_keys]
            return torch.tensor(arr, dtype=torch.float32)
        return None

    def __getitem__(self, idx):
        r = self.records[idx]
        text = (r.get("transcript") or r.get("text") or "").strip()
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        audio_vec = self._extract_audio_vec(r)
        video_vec = self._extract_video_vec(r)
        phq = r.get("phq_score") or r.get("phq") or r.get("target_phq") or r.get("label_phq")
        try:
            phq_val = float(phq) if phq is not None else 0.0
        except Exception:
            phq_val = 0.0
        label = 1 if phq_val > 10.0 else 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_vec": audio_vec,
            "video_vec": video_vec,
            "phq": torch.tensor(phq_val, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }

def collate_batch(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)

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

# -------------------------
# Model components
# -------------------------
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
        self.phq_mu = nn.Linear(hidden, 1)
        self.phq_logsigma = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.drop(self.act(self.fc1(x)))
        logits = self.classifier(h)
        mu = self.phq_mu(h).squeeze(-1)
        log_sigma = self.phq_logsigma(h).squeeze(-1)
        return logits, mu, log_sigma

class MultiModalModel(nn.Module):
    def __init__(self, bert_name: str, audio_dim: Optional[int], vision_dim: Optional[int], device: str = "cpu"):
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
        if audio_enc is not None: parts.append(audio_enc)
        if vision_enc is not None: parts.append(vision_enc)
        fused = torch.cat(parts, dim=1)
        logits, mu, log_sigma = self.fusion(fused)
        reg_pred = mu
        if rl_mode and sample_action:
            sigma = torch.exp(log_sigma).clamp(min=1e-4)
            noise = torch.randn_like(mu)
            action = mu + sigma * noise
            var = sigma * sigma
            log_prob = -0.5 * (((action - mu) ** 2) / var + 2 * log_sigma + math.log(2 * math.pi))
            return logits, reg_pred, (action, log_prob)
        else:
            return logits, reg_pred, (mu, log_sigma)

# -------------------------
# Training / utils
# -------------------------
def run_inference(model: MultiModalModel, dataloader: DataLoader, device: str = "cpu"):
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

def fine_tune_supervised(model: MultiModalModel, dataset: MultiModalDataset, epochs: int = 1, batch_size: int = 8, lr: float = 2e-5, device: str = "cpu", val_dataset: Optional[MultiModalDataset] = None):
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    optimizer = AdamW(model.parameters(), lr=lr)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; steps = 0
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
            total_loss += loss.item(); steps += 1
        avg = total_loss/steps if steps else 0.0
        print(f"[supervised] epoch {epoch+1}/{epochs} avg_loss={avg:.4f}")
        # optional small validation at epoch end
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
            preds = run_inference(model, val_loader, device=device)
            y_true = []
            for rec in val_dataset.records:
                phq = rec.get("phq_score") or rec.get("phq") or rec.get("target_phq")
                try:
                    y_true.append(float(phq))
                except:
                    y_true.append(0.0)
            # compute simple MAE and binary metrics
            pred_phq = np.array([p["pred_phq"] for p in preds][:len(y_true)])
            mae = mean_absolute_error(np.array(y_true), pred_phq) if len(y_true) and len(pred_phq) else None
            print(f"[val] MAE={mae}")
    return model

def compute_state_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    delta = {}
    for k in after:
        if k in before and before[k].shape == after[k].shape:
            delta[k] = (after[k].detach().cpu() - before[k].detach().cpu()).clone()
        else:
            delta[k] = after[k].detach().cpu().clone()
    return delta

def apply_safety_to_delta(delta: Dict[str, torch.Tensor], max_param_change: float = 1e-3, max_global_norm: float = 1.0) -> Dict[str, torch.Tensor]:
    # per-parameter clamp
    for k in list(delta.keys()):
        t = delta[k]
        delta[k] = t.clamp(min=-max_param_change, max=max_param_change)
    # global norm
    total_sq = 0.0
    for k in delta:
        total_sq += (delta[k].float().norm() ** 2).item()
    global_norm = math.sqrt(total_sq)
    if global_norm > max_global_norm:
        scale = max_global_norm / (global_norm + 1e-12)
        for k in delta:
            delta[k] = (delta[k].float() * scale).clone()
    return delta

def save_model_and_delta(model: MultiModalModel, base_state: Dict[str, torch.Tensor], out_path: str, delta_out_path: str, store_root: str):
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # save model state
    sd = model.state_dict()
    torch.save(sd, out_path)
    print(f"[SAVE] model state saved -> {out_path}")
    # compute delta vs base
    after = {k: v.detach().cpu().clone() for k, v in sd.items()}
    delta = compute_state_delta(base_state, after)
    # safety
    delta_safe = apply_safety_to_delta(delta, max_param_change=1e-3, max_global_norm=1.0)
    # save delta as .pt
    delta_buf_path = Path(delta_out_path)
    delta_buf_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta_safe, str(delta_buf_path))
    # receipt (simple)
    receipt = {
        "agent": "trainer-agent",
        "operation": "local_update",
        "timestamp": time.time(),
        "outputs": [str(delta_buf_path)],
        "params": {"max_param_change": 1e-3, "max_global_norm": 1.0}
    }
    receipt_path = str(delta_buf_path.with_suffix(".receipt.json"))
    with open(receipt_path, "w") as f:
        json.dump(receipt, f)
    # also write to secure store location
    store = SecureStoreFallback(root=store_root)
    try:
        uri = store.encrypt_write(f"file://{Path(store_root)/delta_buf_path.name}", delta_buf_path.read_bytes())
        print(f"[STORE] delta written to store -> {uri}")
    except Exception as e:
        print(f"[WARN] failed to write delta to secure store: {e}")
    return str(delta_buf_path), receipt_path

# -------------------------
# Explainability functions
# -------------------------
def modality_ablation_importance(model: MultiModalModel, sample_batch: Dict[str, torch.Tensor], device: str = "cpu"):
    model.eval()
    with torch.no_grad():
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample_batch.items()}
        logits, reg, _ = model(inputs["input_ids"], inputs["attention_mask"], audio_vec=inputs.get("audio_vec"), vision_vec=inputs.get("video_vec"))
        base_pos = float(torch.softmax(logits, dim=1)[:, 1].mean().item())
        base_phq = float(reg.mean().item())
        results = {}
        # audio
        if inputs.get("audio_vec") is not None:
            zero_a = torch.zeros_like(inputs["audio_vec"])
            l_a, r_a, _ = model(inputs["input_ids"], inputs["attention_mask"], audio_vec=zero_a, vision_vec=inputs.get("video_vec"))
            p_a = float(torch.softmax(l_a, dim=1)[:, 1].mean().item())
            r_a_val = float(r_a.mean().item())
            results["audio_posdelta"] = abs(base_pos - p_a)
            results["audio_phqdelta"] = abs(base_phq - r_a_val)
        else:
            results["audio_posdelta"] = 0.0; results["audio_phqdelta"] = 0.0
        # vision
        if inputs.get("video_vec") is not None:
            zero_v = torch.zeros_like(inputs["video_vec"])
            l_v, r_v, _ = model(inputs["input_ids"], inputs["attention_mask"], audio_vec=inputs.get("audio_vec"), vision_vec=zero_v)
            p_v = float(torch.softmax(l_v, dim=1)[:, 1].mean().item())
            r_v_val = float(r_v.mean().item())
            results["vision_posdelta"] = abs(base_pos - p_v)
            results["vision_phqdelta"] = abs(base_phq - r_v_val)
        else:
            results["vision_posdelta"] = 0.0; results["vision_phqdelta"] = 0.0
        # text ablation via empty string
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name) if "args" in globals() else None
            if tokenizer:
                empty = tokenizer([""] * inputs["input_ids"].shape[0], padding=True, truncation=True, return_tensors="pt")
                empty_ids = empty["input_ids"].to(device); empty_mask = empty["attention_mask"].to(device)
                l_t, r_t, _ = model(empty_ids, empty_mask, audio_vec=inputs.get("audio_vec"), vision_vec=inputs.get("video_vec"))
                p_t = float(torch.softmax(l_t, dim=1)[:, 1].mean().item())
                r_t_val = float(r_t.mean().item())
                results["text_posdelta"] = abs(base_pos - p_t)
                results["text_phqdelta"] = abs(base_phq - r_t_val)
            else:
                results["text_posdelta"] = 0.0; results["text_phqdelta"] = 0.0
        except Exception:
            results["text_posdelta"] = 0.0; results["text_phqdelta"] = 0.0
        agg = {
            "audio_score": (results["audio_posdelta"] + results["audio_phqdelta"]) / 2.0,
            "vision_score": (results["vision_posdelta"] + results["vision_phqdelta"]) / 2.0,
            "text_score": (results["text_posdelta"] + results["text_phqdelta"]) / 2.0,
            "raw": results
        }
        return agg

def explain_linear_probe(clf, X_orig, X_new, y, out_dir, prefix="probe"):
    os.makedirs(out_dir, exist_ok=True)
    fname = Path(out_dir) / f"{prefix}_explain_{int(time.time()*1000)}.txt"
    lines = []
    try:
        coefs = getattr(clf, "coef_", None)
        if coefs is None:
            lines.append("No linear coefficients available.")
        else:
            coefs = np.array(coefs)
            lines.append(f"coef_shape={coefs.shape}")
            if coefs.ndim == 2:
                avg = np.mean(np.abs(coefs), axis=0)
            else:
                avg = np.abs(coefs).reshape(-1)
            top = np.argsort(-avg)[:30].tolist()
            lines.append(f"Top global features: {top}")
            for i in range(min(20, X_new.shape[0])):
                xi = X_new[i]
                if coefs.ndim == 2 and coefs.shape[0] > 1:
                    try:
                        pred = int(clf.predict(xi.reshape(1, -1))[0])
                        row = coefs[pred]
                    except Exception:
                        row = coefs.mean(axis=0)
                else:
                    row = coefs.reshape(-1)
                contrib = row * xi
                top_idx = np.argsort(-np.abs(contrib))[:10].tolist()
                lines.append(f"sample{i} pred={clf.predict(xi.reshape(1,-1))[0]} top_contrib_idx={top_idx}")
    except Exception as e:
        lines.append(f"Explain failed: {repr(e)}")
    with open(fname, "w") as f:
        f.write("\n".join(lines))
    return str(fname)

# -------------------------
# CLI / Orchestration
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Trainer MentalBERT (DAIC-WOZ style) - supervised + explainability")
    p.add_argument("--input", required=True, help="Path to parquet/json/csv with session records")
    p.add_argument("--mode", choices=["autonomous", "supervised", "rl"], default="supervised")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--out-path", default="./trainer_outputs/mentalbert_privacy_subset.pt")
    p.add_argument("--delta-path", default="./trainer_outputs/mentalbert_delta.pt")
    p.add_argument("--store-root", default="./trainer_outputs/secure_store")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--binarize", action="store_true", help="Compute classification metrics on PHQ>threshold")
    p.add_argument("--binarize-threshold", type=float, default=10.0)
    p.add_argument("--bert-model-name", default="mental/mental-bert-base-uncased")
    p.add_argument("--max-param-change", type=float, default=1e-3)
    p.add_argument("--max-global-delta-norm", type=float, default=1.0)
    return p.parse_args()

def main():
    global args  # used in modality_ablation_importance for tokenizer retrieval
    args = parse_args()
    device = args.device

    print("[INFO] loading records from", args.input)
    records = read_parquet_records(args.input)
    print(f"[INFO] loaded {len(records)} records")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    ds = MultiModalDataset(records, tokenizer, max_len=128)

    # infer audio/video dims
    audio_dim = None; vision_dim = None
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "audio" in f:
            a = f["audio"]
            if isinstance(a, dict) and a.get("wav2vec2"):
                audio_dim = len(a["wav2vec2"]); break
            if isinstance(a, dict) and a.get("egemaps"):
                audio_dim = len(a["egemaps"].keys()); break
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "video" in f and isinstance(f["video"], dict) and f["video"].get("densenet"):
            vision_dim = len(f["video"]["densenet"]); break
        if any(k.startswith("neuron_") for k in r.keys()):
            neuron_keys = sorted([kk for kk in r.keys() if kk.startswith("neuron_")])
            vision_dim = len(neuron_keys); break

    print(f"[INFO] inferred dims audio_dim={audio_dim} vision_dim={vision_dim}")

    model = MultiModalModel(args.bert_model_name, audio_dim=audio_dim, vision_dim=vision_dim, device=device)
    model.to(device)

    # Save base state for delta computation
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # split train/val
    n = len(ds)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    vs = int(n * args.val_split) if n > 1 else 0
    val_idx = idx[:vs].tolist()
    train_idx = idx[vs:].tolist()

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx] if val_idx else None
    train_ds = MultiModalDataset(train_records, tokenizer, max_len=128)
    val_ds = MultiModalDataset(val_records, tokenizer, max_len=128) if val_records is not None else None

    print(f"[INFO] train={len(train_records)} val={len(val_records) if val_records is not None else 0}")

    if args.mode == "autonomous":
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        preds = run_inference(model, loader, device=device)
        # save preds
        out_pred_csv = Path(args.out_path).parent / "inference_preds.csv"
        pd.DataFrame(preds).to_csv(out_pred_csv, index=False)
        # modality ablation on first batch
        first_batch = None
        for b in loader:
            first_batch = b
            break
        explanations = modality_ablation_importance(model, first_batch, device=device) if first_batch else {}
        out_json = Path(args.out_path).parent / "inference_explain.json"
        with open(out_json, "w") as f:
            json.dump({"explainability": explanations, "pred_count": len(preds)}, f)
        print("[DONE] autonomous inference written:", out_pred_csv, out_json)
        return

    elif args.mode in ("supervised", "rl"):
        # supervised fine-tune
        print("[TRAIN] starting supervised fine-tune")
        model = fine_tune_supervised(model, train_ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, val_dataset=val_ds)
        # evaluate on validation (or train if no val)
        eval_ds = val_ds if val_ds is not None else train_ds
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        preds = run_inference(model, eval_loader, device=device)
        y_true = []
        for rec in eval_ds.records:
            phq = rec.get("phq_score") or rec.get("phq") or rec.get("target_phq")
            try:
                y_true.append(float(phq))
            except:
                y_true.append(0.0)
        # compute metrics
        pred_phq = np.array([p["pred_phq"] for p in preds][:len(y_true)])
        mae = mean_absolute_error(np.array(y_true), pred_phq) if len(y_true) and len(pred_phq) else None
        print(f"[EVAL] regression MAE: {mae}")

        if args.binarize:
            y_bin = (np.array(y_true) > args.binarize_threshold).astype(int)
            pred_class = np.array([p["pred_class"] for p in preds][:len(y_bin)])
            acc = accuracy_score(y_bin, pred_class) if len(y_bin) else None
            prec = precision_score(y_bin, pred_class, zero_division=0) if len(y_bin) else None
            rec = recall_score(y_bin, pred_class, zero_division=0) if len(y_bin) else None
            f1 = f1_score(y_bin, pred_class, zero_division=0) if len(y_bin) else None
            print(f"[EVAL] classification acc={acc} prec={prec} rec={rec} f1={f1}")
        else:
            acc = prec = rec = f1 = None

        # save per-sample preds + explain summary
        out_dir = Path(args.out_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        preds_csv = out_dir / "eval_preds.csv"
        pd.DataFrame(preds).to_csv(preds_csv, index=False)
        # modality ablation on a sample batch (first)
        sample_batch = None
        for b in eval_loader:
            sample_batch = b
            break
        mod_exp = modality_ablation_importance(model, sample_batch, device=device) if sample_batch else {}
        with open(out_dir / "modality_ablation.json", "w") as f:
            json.dump(mod_exp, f)
        # linear-probe explainability on final embedding
        # Build pooled embeddings for explain probe
        model.eval()
        embs = []
        for b in eval_loader:
            with torch.no_grad():
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
                bert_out = model.bert(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                pooled = bert_out.last_hidden_state[:, 0, :].cpu().numpy()
                # optionally append audio/video enc outputs if present
                if model.has_audio and inputs.get("audio_vec") is not None:
                    aenc = model.audio_encoder(inputs["audio_vec"].to(device)).cpu().numpy()
                    pooled = np.concatenate([pooled, aenc], axis=1)
                if model.has_vision and inputs.get("video_vec") is not None:
                    venc = model.vision_encoder(inputs["video_vec"].to(device)).cpu().numpy()
                    pooled = np.concatenate([pooled, venc], axis=1)
                embs.append(pooled)
        if embs:
            X_emb = np.vstack(embs)
            y_vals = np.array([float(r.get("phq_score") or r.get("phq") or 0.0) for r in eval_ds.records])[: X_emb.shape[0]]
            # train a linear probe for explanation
            try:
                if args.binarize:
                    y_cls = (y_vals > args.binarize_threshold).astype(int)
                    if len(np.unique(y_cls)) > 1:
                        clf = LogisticRegression(max_iter=1000, solver="liblinear")
                        clf.fit(X_emb, y_cls)
                        explain_linear_probe(clf, X_emb, X_emb, y_cls, out_dir=str(out_dir), prefix="final_probe")
                else:
                    reg = LinearRegression()
                    reg.fit(X_emb, y_vals)
                    explain_linear_probe(reg, X_emb, X_emb, y_vals, out_dir=str(out_dir), prefix="final_probe_reg")
            except Exception as e:
                print("[WARN] linear probe explain failed:", e)

        # Save model and delta
        delta_path, receipt_path = save_model_and_delta(model, base_state, args.out_path, args.delta_path, args.store_root)

        # final report
        report = {
            "model_path": str(Path(args.out_path).absolute()),
            "delta_path": delta_path,
            "delta_receipt": receipt_path,
            "metrics": {"mae": mae, "acc": acc, "precision": prec, "recall": rec, "f1": f1},
            "preds_csv": str(preds_csv),
            "modality_ablation": str(out_dir / "modality_ablation.json")
        }
        report_path = out_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print("[DONE] training complete. Report written to:", report_path)
        return

    else:
        raise ValueError("unknown mode")

if __name__ == "__main__":
    main()
