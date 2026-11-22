"""VisemeTemporalNet: Conv + BiLSTM + scaled dot-product self-attention, with built-in feature normalisation."""
import os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import lip_features as LF
except ImportError:
    import lip_features as LF

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(MODELS_DIR, "lip_model.pth")   # plain state_dict (incl. normalisation buffers)
META_PATH = os.path.join(MODELS_DIR, "lip_model_meta.json")  # classes, input_dim, feature format, run_id ...

VOCABULARY = [
    "RESTING",
    "HELP",
    "WATER",
    "YES",
    "NO",
    "DOCTOR",
    "PAIN",
    "THANK YOU",
    "TIRED",
    "NURSE",
    "BREATHE",
    "MEDICINE"
]
VOCAB_MAP = {word: idx for idx, word in enumerate(VOCABULARY)}


class ConvBlock(nn.Module):
    def __init__(self, ch, k=3, p=0.1):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.bn = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(F.gelu(self.bn(self.conv(x))))


class VisemeTemporalNet(nn.Module):
    def __init__(self, num_classes=len(VOCABULARY), input_dim=12, hidden=64, layers=2, dropout=0.3, conv_ch=64, **kwargs):
        super().__init__()
        # Allow hidden_dim alias for backward compatibility
        if "hidden_dim" in kwargs:
            hidden = kwargs["hidden_dim"]
        self.num_classes, self.input_dim = num_classes, input_dim
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))
        self.proj = nn.Sequential(nn.Conv1d(input_dim, conv_ch, 3, padding=1), nn.BatchNorm1d(conv_ch), nn.GELU())
        self.blocks = nn.Sequential(ConvBlock(conv_ch), ConvBlock(conv_ch))
        self.lstm = nn.LSTM(conv_ch, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        d = hidden * 2
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d * 3)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d * 3, num_classes)

    def forward(self, x):                      # x: (B, T, D) raw features
        x = (x - self.feat_mean) / self.feat_std
        h = self.blocks(self.proj(x.transpose(1, 2))).transpose(1, 2)   # (B, T, C)
        out, _ = self.lstm(h)                                           # (B, T, 2H)
        att = torch.softmax(self.q(out) @ self.k(out).transpose(1, 2) / math.sqrt(out.size(-1)), dim=-1)
        ctx = att @ self.v(out)
        z = torch.cat([ctx.mean(1), out.mean(1), out.max(1).values], dim=-1)
        logits = self.head(self.drop(self.norm(z)))
        return logits


def load_model(models_dir=MODELS_DIR, device="cpu"):
    """Returns (model, meta) or (None, None) if nothing has been trained yet."""
    mp, wp = os.path.join(models_dir, "lip_model_meta.json"), os.path.join(models_dir, "lip_model.pth")
    if not (os.path.exists(mp) and os.path.exists(wp)):
        return None, None
    try:
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        m = VisemeTemporalNet(meta["num_classes"], meta["input_dim"], meta.get("hidden", 64),
                              meta.get("layers", 2), meta.get("dropout", 0.3))
        state = torch.load(wp, map_location=device)
        m.load_state_dict(state)
        return m.to(device).eval(), meta
    except Exception as e:
        print(f"[LOAD_MODEL WARNING] Could not load model: {e}")
        return None, None


@torch.no_grad()
def predict(model, meta, arr, device="cpu"):
    """arr: (T,12) legacy or (T,44,2|3) lip landmarks. Returns (probs ndarray, best_class_name)."""
    arr = np.asarray(arr, dtype=np.float32)
    expected_fmt = meta.get("feature_format", "legacy12") if meta else "legacy12"
    seq_len = meta.get("seq_len", 30) if meta else 30

    feats = LF.prepare_features(arr, expected_fmt, seq_len)
    in_dim = getattr(model, "input_dim", feats.shape[-1])
    if feats.shape[-1] != in_dim:
        if feats.shape[-1] > in_dim:
            feats = feats[:, :in_dim]
        else:
            pad = np.zeros((feats.shape[0], in_dim - feats.shape[-1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=-1)

    x = torch.from_numpy(feats).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    classes = meta.get("classes", VOCABULARY) if meta else VOCABULARY
    pred_idx = int(probs.argmax())
    best_class = classes[pred_idx] if pred_idx < len(classes) else "UNKNOWN"
    return probs, best_class
