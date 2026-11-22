"""
Local training engine.   python models/train_custom.py --epochs 100

What changed vs. before
  * Epochs are real and honest: exactly `epochs` passes are run (patience=0 default), each logged 1..N.
    One epoch = one pass over the training set, each sample seen `aug_copies` times with FRESH augmentation.
  * Scheduler steps once per EPOCH (warm-up + cosine), best-validation checkpoint is restored at the end.
  * Split happens BEFORE augmentation (no leakage), stratified per phrase.
  * Everything is recorded in models/runs/<run_id>/ and models/runs_index.csv.
"""
import os, sys, json, time, csv, copy, argparse, random, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from . import lip_features as LF, dataset_manager as DM
    from .lip_model import VisemeTemporalNet, MODELS_DIR
except ImportError:
    import lip_features as LF, dataset_manager as DM
    from lip_model import VisemeTemporalNet, MODELS_DIR

RUNS_DIR = os.path.join(MODELS_DIR, "runs")
STATUS_PATH = os.path.join(MODELS_DIR, "train_status.json")


# ---------------------------------------------------------------- helpers
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def _dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_Enc)
    os.replace(tmp, path)


def _time_sample(x, src):
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, x.shape[0] - 1)
    w = (src - lo).reshape((-1,) + (1,) * (x.ndim - 1)).astype(np.float32)
    return x[lo] * (1 - w) + x[hi] * w


def augment_raw(arr, fmt, rng, noise_std=None):
    """Time-warp + shift (both formats); noise/scale at the right level for each format."""
    T = arr.shape[0]
    knots = np.linspace(0, 1, 5)
    speeds = np.clip(1 + 0.3 * rng.standard_normal(4), 0.4, 1.8)
    cum = np.concatenate([[0], np.cumsum(speeds)]); cum /= cum[-1]
    src = np.interp(np.linspace(0, 1, T), knots, cum) * (T - 1) + rng.uniform(-2, 2)
    x = _time_sample(arr, np.clip(src, 0, T - 1))
    if fmt == "legacy12":
        x = x * rng.uniform(0.9, 1.1)
        if noise_std is not None:
            x = x + rng.normal(0, 1, x.shape) * 0.05 * noise_std
    else:
        xy = x[..., :2]
        width = max(float(np.median(np.linalg.norm(xy[:, 10] - xy[:, 0], axis=-1))), 1e-6)
        mid = xy.mean(axis=(0, 1), keepdims=True)
        x[..., :2] = (xy - mid) * rng.uniform(0.95, 1.05, size=2) + mid          # mild anisotropic face-shape change
        x[..., :2] += rng.normal(0, 0.008 * width, x[..., :2].shape)            # landmark jitter
    return x.astype(np.float32)


class LipDataset(Dataset):
    def __init__(self, arrays, labels, fmt, seq_len, augment, copies=1, noise_std=None, seed=0):
        self.arrays, self.labels, self.fmt, self.seq_len = arrays, labels, fmt, seq_len
        self.augment, self.copies, self.noise_std, self.seed, self.epoch = augment, copies, noise_std, seed, 0

    def set_epoch(self, e): self.epoch = e
    def __len__(self): return len(self.arrays) * (self.copies if self.augment else 1)

    def __getitem__(self, i):
        idx = i % len(self.arrays)
        arr = self.arrays[idx]
        if self.augment:
            rng = np.random.default_rng(self.seed * 1_000_003 + self.epoch * 10_007 + i)
            arr = augment_raw(arr, self.fmt, rng, self.noise_std)
        f = LF.prepare_features(arr, self.fmt, self.seq_len)
        return torch.from_numpy(f), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def stratified_split(labels, val_frac, rng):
    tr, va = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]; rng.shuffle(idx)
        n_val = 0 if (val_frac <= 0 or len(idx) < 4) else max(1, int(round(len(idx) * val_frac)))
        va += idx[:n_val].tolist(); tr += idx[n_val:].tolist()
    return np.array(sorted(tr)), np.array(sorted(va), dtype=int)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); tot, n, ys, ps = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        lo = model(x)
        tot += criterion(lo, y).item() * len(y); n += len(y)
        ys += y.cpu().tolist(); ps += lo.argmax(1).cpu().tolist()
    ys, ps = np.array(ys), np.array(ps)
    return tot / max(n, 1), float((ys == ps).mean()) if n else 0.0, ys, ps


def class_report(y, p, classes):
    C = len(classes); cm = np.zeros((C, C), int)
    for t, q in zip(y, p): cm[t, q] += 1
    per = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                  "support": int(cm[i].sum())}
    conf = sorted([(int(cm[i, j]), classes[i], classes[j]) for i in range(C) for j in range(C) if i != j and cm[i, j]],
                  reverse=True)[:8]
    return cm, per, [{"count": a, "true": b, "predicted_as": c} for a, b, c in conf]


# ---------------------------------------------------------------- main entry
def train(epochs=100, batch_size=16, lr=2e-3, weight_decay=1e-2, val_frac=0.2, aug_copies=4, patience=0,
          label_smoothing=0.1, hidden=64, layers=2, dropout=0.3, warmup=5, seed=42, fmt="auto",
          include_seed=False, seq_len=30, device=None, progress_cb=None):
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id); os.makedirs(run_dir, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True); log_lines.append(msg)

    def status(**kw):
        _dump({"run_id": run_id, "epochs": epochs, "elapsed_s": round(time.time() - t0, 1), **kw}, STATUS_PATH)

    try:
        status(state="loading")
        try:
            data = DM.load_dataset(fmt, seq_len, include_seed)
        except ValueError:
            # Try falling back to include_seed=True if standard dataset was empty
            data = DM.load_dataset(fmt, seq_len, include_seed=True)

        classes, labels, arrays, fmt = data["classes"], data["labels"], data["arrays"], data["fmt"]
        if len(classes) < 2:
            raise ValueError("need at least 2 phrases with recorded samples")
        if data["dropped_other_format"]:
            log(f"WARNING: ignored {data['dropped_other_format']} samples in a different format than '{fmt}'")
        log(f"run {run_id} | format={fmt} | classes={len(classes)} | samples={len(arrays)} | device={device}")
        log(f"samples per phrase: {data['counts']}")

        rng = np.random.default_rng(seed)
        tr_idx, va_idx = stratified_split(labels, val_frac, rng)
        has_val = len(va_idx) > 0
        if not has_val:
            log("WARNING: not enough samples per phrase for a validation split (need >=4). Record more samples "
                "- accuracy below is TRAIN accuracy only.")
        tr_arr, tr_lab = [arrays[i] for i in tr_idx], labels[tr_idx]
        va_arr, va_lab = [arrays[i] for i in va_idx], labels[va_idx]

        _dump({"format": fmt, "classes": classes, "counts": data["counts"],
               "files": [{"file": f, "class": classes[l], "split": "val" if i in set(va_idx.tolist()) else "train",
                          "sha1": DM.sha1_file(os.path.join(DM.ROOT, f))}
                         for i, (f, l) in enumerate(zip(data["files"], labels))]},
              os.path.join(run_dir, "dataset_manifest.json"))

        feats = np.concatenate([LF.prepare_features(a, fmt, seq_len) for a in tr_arr], 0)
        mean, std = feats.mean(0), feats.std(0)
        std = np.maximum(std, 0.05 * np.median(std) + 1e-3)

        train_ds = LipDataset(tr_arr, tr_lab, fmt, seq_len, True, aug_copies, std if fmt == "legacy12" else None, seed)
        drop_last = len(train_ds) > batch_size and len(train_ds) % batch_size == 1
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        eval_train_dl = DataLoader(LipDataset(tr_arr, tr_lab, fmt, seq_len, False), batch_size=64)
        val_dl = DataLoader(LipDataset(va_arr, va_lab, fmt, seq_len, False), batch_size=64) if has_val else None

        in_dim = LF.feature_dim(fmt)
        model = VisemeTemporalNet(len(classes), in_dim, hidden, layers, dropout).to(device)
        model.feat_mean.copy_(torch.from_numpy(mean)); model.feat_std.copy_(torch.from_numpy(std))

        cnt = np.bincount(tr_lab, minlength=len(classes)).astype(np.float32)
        cw = torch.tensor(cnt.sum() / (len(classes) * np.maximum(cnt, 1)), dtype=torch.float32).to(device)
        crit_train = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
        crit_eval = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if warmup > 0 and epochs > warmup + 1:
            sched = torch.optim.lr_scheduler.SequentialLR(opt, [
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup, eta_min=lr * 0.01)],
                milestones=[warmup])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.01)

        config = dict(epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay, val_frac=val_frac,
                      aug_copies=aug_copies, patience=patience, label_smoothing=label_smoothing, hidden=hidden,
                      layers=layers, dropout=dropout, warmup=warmup, seed=seed, format=fmt, seq_len=seq_len,
                      device=device, train_samples=len(tr_arr), val_samples=len(va_arr),
                      samples_per_epoch=len(train_ds), steps_per_epoch=len(train_dl))
        _dump(config, os.path.join(run_dir, "config.json"))
        log(f"epoch = 1 pass over {len(tr_arr)} train samples x {aug_copies} fresh augmentations "
            f"= {len(train_ds)} samples, {len(train_dl)} steps")

        history, best_key, best_epoch, best_state, bad = [], None, 0, None, 0
        for ep in range(1, epochs + 1):
            te = time.time(); train_ds.set_epoch(ep); model.train(); tl, tn, tc = 0.0, 0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                lo = model(x); loss = crit_train(lo, y)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tl += loss.item() * len(y); tn += len(y); tc += (lo.argmax(1) == y).sum().item()
            lr_now = opt.param_groups[0]["lr"]; sched.step()
            vl, va_acc = (None, None)
            if has_val:
                vl, va_acc, _, _ = evaluate(model, val_dl, crit_eval, device)
            row = {"epoch": ep, "train_loss": tl / tn, "train_acc": tc / tn, "val_loss": vl, "val_acc": va_acc,
                   "lr": lr_now, "seconds": round(time.time() - te, 3)}
            history.append(row)
            key = (va_acc, -vl) if has_val else (-row["train_loss"],)
            if best_key is None or key > best_key:
                best_key, best_epoch, bad = key, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            log(f"epoch {ep:03d}/{epochs} | loss {row['train_loss']:.4f} acc {row['train_acc']:.3f}"
                + (f" | val_loss {vl:.4f} val_acc {va_acc:.3f}" if has_val else "")
                + f" | lr {lr_now:.2e} | {row['seconds']:.2f}s" + (" *best" if best_epoch == ep else ""))
            info = dict(state="training", epoch=ep, best_epoch=best_epoch, **{k: row[k] for k in
                        ("train_loss", "train_acc", "val_loss", "val_acc")})
            status(**info)
            if progress_cb: progress_cb(info)
            if patience and bad >= patience:
                log(f"early stop at epoch {ep} (no improvement for {patience} epochs)"); break

        epochs_done = len(history)
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        rep_dl, rep_name = (val_dl, "val") if has_val else (eval_train_dl, "train")
        _, rep_acc, ys, ps = evaluate(model, rep_dl, crit_eval, device)
        _, tr_acc, _, _ = evaluate(model, eval_train_dl, crit_eval, device)
        cm, per, conf = class_report(ys, ps, classes)

        # ---- saved records
        torch.save(best_state, os.path.join(run_dir, "best.pth"))
        torch.save(last_state, os.path.join(run_dir, "last.pth"))
        with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
        _dump(history, os.path.join(run_dir, "history.json"))
        metrics = {"run_id": run_id, "epochs_requested": epochs, "epochs_completed": epochs_done,
                   "best_epoch": best_epoch, "report_split": rep_name, f"{rep_name}_accuracy": rep_acc,
                   "train_accuracy": tr_acc, "per_class": per, "top_confusions": conf,
                   "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
                   "total_seconds": round(time.time() - t0, 1)}
        _dump(metrics, os.path.join(run_dir, "metrics.json"))
        with open(os.path.join(run_dir, "train_log.txt"), "w") as f: f.write("\n".join(log_lines))

        meta = {"run_id": run_id, "classes": classes, "num_classes": len(classes), "input_dim": in_dim,
                "feature_format": fmt, "seq_len": seq_len, "hidden": hidden, "layers": layers, "dropout": dropout,
                "epochs_completed": epochs_done, "best_epoch": best_epoch, f"{rep_name}_accuracy": rep_acc}
        tmp = os.path.join(MODELS_DIR, "lip_model.pth.tmp")
        torch.save(best_state, tmp); os.replace(tmp, os.path.join(MODELS_DIR, "lip_model.pth"))   # atomic -> safe hot-reload
        _dump(meta, os.path.join(MODELS_DIR, "lip_model_meta.json"))

        # Also export classes to model_classes.json and weights to lip_model_weights.npz for numpy inference compatibility
        try:
            with open(os.path.join(MODELS_DIR, "model_classes.json"), "w", encoding="utf-8") as f:
                json.dump(classes, f, indent=2)
            np_weights = {k: v.cpu().numpy() for k, v in best_state.items()}
            np.savez(os.path.join(MODELS_DIR, "lip_model_weights.npz"), **np_weights, model_classes=np.array(classes))
        except Exception as e:
            log(f"NOTE: could not export .npz: {e}")

        idx = os.path.join(MODELS_DIR, "runs_index.csv"); new = not os.path.exists(idx)
        with open(idx, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(["run_id", "format", "classes", "samples", "epochs_completed", "best_epoch",
                                f"{rep_name}_accuracy", "train_accuracy", "seconds"])
            w.writerow([run_id, fmt, len(classes), len(arrays), epochs_done, best_epoch, f"{rep_acc:.4f}",
                        f"{tr_acc:.4f}", metrics["total_seconds"]])
        log(f"DONE: {epochs_done}/{epochs} epochs | best epoch {best_epoch} | {rep_name} acc {rep_acc:.3f} | saved -> {run_dir}")
        if conf: log(f"most confused: {conf[:3]}")
        status(state="done", epoch=epochs_done, best_epoch=best_epoch, **{f"{rep_name}_accuracy": rep_acc})
        return {**metrics, "run_dir": run_dir, "meta": meta}
    except Exception as e:
        status(state="error", error=str(e)); raise


train_custom_model = train_model = train_custom_100_epochs = train


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--aug-copies", type=int, default=4)
    ap.add_argument("--patience", type=int, default=0, help="0 = always run all epochs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--format", default="auto", choices=["auto", "legacy12", "landmarks44"])
    ap.add_argument("--include-seed", action="store_true", help="also train on generated baseline/seed files")
    a = ap.parse_args()
    train(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay, val_frac=a.val_frac,
          aug_copies=a.aug_copies, patience=a.patience, seed=a.seed, fmt=a.format, include_seed=a.include_seed)
