"""Disk-backed dataset: dataset/<PHRASE>/sample_<ms>.npy  +  dataset/records.csv (saved record of every sample)."""
import os, re, csv, time, hashlib, shutil
import numpy as np

try:
    from . import lip_features as LF
except ImportError:
    import lip_features as LF

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODELS_DIR)

# Vercel Serverless environment handling: write operations must target /tmp
IS_VERCEL = bool(os.environ.get("VERCEL"))
if IS_VERCEL:
    DATASET_DIR = "/tmp/dataset"
    VOCAB_FILE = "/tmp/vocab.json"
else:
    DATASET_DIR = os.path.join(ROOT, "dataset")
    VOCAB_FILE = os.path.join(MODELS_DIR, "vocab.json")

RECORDS_CSV = os.path.join(DATASET_DIR, "records.csv")

DEFAULT_VOCABULARY = [
    "RESTING",
    "HELP",
    "WATER",
    "DOCTOR",
    "PAIN",
    "PRINCE KUMAR",
    "THANK YOU",
    "YES",
    "NO",
    "BREATHE",
    "NURSE",
    "MEDICINE",
    "TIRED",
    "I WANT WATER"
]


def sanitize_phrase(phrase):
    s = re.sub(r"[^A-Za-z0-9 _-]", "", str(phrase)).strip().upper()
    s = re.sub(r"[ -]+", "_", s)
    if not s:
        raise ValueError("empty phrase")
    return s

clean_folder_name = sanitize_phrase


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def add_phrase(phrase):
    name = sanitize_phrase(phrase)
    os.makedirs(os.path.join(DATASET_DIR, name), exist_ok=True)
    return name

add_custom_phrase = add_phrase


def list_classes():
    if not os.path.isdir(DATASET_DIR):
        return []
    return sorted(d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)))


def get_dynamic_vocabulary():
    classes = list_classes()
    if classes:
        # Guarantee RESTING is first if present
        if "RESTING" in classes:
            classes.remove("RESTING")
            return ["RESTING"] + classes
        return classes
    if os.path.exists(VOCAB_FILE):
        try:
            import json
            with open(VOCAB_FILE, "r", encoding="utf-8") as f:
                v = json.load(f)
                if isinstance(v, list) and len(v) > 0:
                    return v
        except Exception:
            pass
    return list(DEFAULT_VOCABULARY)


def save_sample(phrase, array, source="studio"):
    """Save one recorded movement and append it to dataset/records.csv. Returns file path."""
    name = add_phrase(phrase)
    arr = np.asarray(array, dtype=np.float32)
    fmt = LF.detect_format(arr)
    if fmt is None:
        raise ValueError(f"unsupported sample shape {arr.shape}; expected (T,12) or (T,44,2|3)")
    stamp = int(time.time() * 1000)
    path = os.path.join(DATASET_DIR, name, f"sample_{stamp}.npy")
    while os.path.exists(path):          # never overwrite an existing recording
        stamp += 1
        path = os.path.join(DATASET_DIR, name, f"sample_{stamp}.npy")
    np.save(path, arr)
    new = not os.path.exists(RECORDS_CSV)
    try:
        with open(RECORDS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "phrase", "file", "shape", "format", "source", "sha1"])
            rel_p = os.path.relpath(path, ROOT) if not IS_VERCEL else path
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, rel_p,
                        "x".join(map(str, arr.shape)), fmt, source, sha1_file(path)])
    except Exception:
        pass
    return path


def save_user_sample(word, tensor, sample_num=None):
    return save_sample(word, tensor, source=f"slot_{sample_num}" if sample_num is not None else "studio")


def clear_phrase_samples(phrase):
    name = sanitize_phrase(phrase)
    folder = os.path.join(DATASET_DIR, name)
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.endswith(".npy"):
                try: os.remove(os.path.join(folder, f))
                except Exception: pass
    return {"status": "success", "word": name}


def reset_entire_dataset():
    if os.path.isdir(DATASET_DIR):
        for d in os.listdir(DATASET_DIR):
            p = os.path.join(DATASET_DIR, d)
            if os.path.isdir(p) and d != "RESTING":
                try: shutil.rmtree(p)
                except Exception: pass
    return {"status": "success"}


def _is_seed(fname):
    low = fname.lower()
    return "seed" in low or "baseline" in low


def load_dataset(fmt="auto", seq_len=30, include_seed=False):
    raw = []  # (arr, class, path, fmt)
    for cls in list_classes():
        d = os.path.join(DATASET_DIR, cls)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".npy") or (not include_seed and _is_seed(fn)):
                continue
            path = os.path.join(d, fn)
            try:
                arr = np.load(path)
            except Exception:
                continue
            f = LF.detect_format(arr)
            if f is not None:
                raw.append((arr.astype(np.float32), cls, path, f))
    if not raw:
        raise ValueError("no usable .npy samples found in dataset/")
    if fmt == "auto":
        fmt = "landmarks44" if any(r[3] == "landmarks44" for r in raw) else "legacy12"
    kept = [r for r in raw if r[3] == fmt]
    dropped = len(raw) - len(kept)
    classes = sorted({r[1] for r in kept})
    cidx = {c: i for i, c in enumerate(classes)}
    return {
        "fmt": fmt,
        "classes": classes,
        "arrays": [LF.resample_time(r[0], seq_len) for r in kept],
        "labels": np.array([cidx[r[1]] for r in kept], dtype=np.int64),
        "files": [os.path.relpath(r[2], ROOT) if not IS_VERCEL else r[2] for r in kept],
        "counts": {c: int(sum(1 for r in kept if r[1] == c)) for c in classes},
        "dropped_other_format": dropped,
    }


def dataset_stats():
    out = {}
    for cls in list_classes():
        d = os.path.join(DATASET_DIR, cls)
        out[cls] = sum(1 for f in os.listdir(d) if f.endswith(".npy"))
    return out


def get_dataset_statistics():
    counts = dataset_stats()
    vocab = get_dynamic_vocabulary()
    for v in vocab:
        if v not in counts:
            counts[v] = 0
    tot = sum(counts.values())
    return {
        "vocabulary": vocab,
        "vocabulary_size": len(vocab),
        "real_recordings": counts,
        "sample_counts": counts,
        "total_recordings": tot,
        "total_samples": tot
    }


class UserKinematicDataset:
    """Wrapper providing backward compatibility for torch training loops."""
    def __init__(self, augment_factor=4, seq_len=30):
        data = load_dataset(fmt="auto", seq_len=seq_len, include_seed=True)
        self.vocabulary = data["classes"]
        self.samples = data["arrays"]
        self.labels = data["labels"]
        self.fmt = data["fmt"]

    def __len__(self):
        return len(self.samples)
