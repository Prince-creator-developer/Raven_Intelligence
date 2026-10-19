import os
import re
import json
import time
import shutil
from typing import Dict, List, Tuple, Optional
import numpy as np

# Optional PyTorch for dataset loading during local training
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH = False
    Dataset = object
    DataLoader = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VOCAB_FILE = os.path.join(BASE_DIR, "models", "vocab.json")

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


def clean_folder_name(phrase: str) -> str:
    """Sanitizes any word or full sentence into a safe folder name."""
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', phrase.strip())
    return sanitized.upper().replace(" ", "_")


def folder_to_display_phrase(folder_name: str) -> str:
    """Converts folder name back to human readable phrase/sentence."""
    return folder_name.replace("_", " ").upper()


def get_dynamic_vocabulary() -> List[str]:
    """
    Scans the dataset directory or loads cached vocab.json.
    100% read-only safe for serverless runtimes.
    """
    vocab_list = []
    if os.path.exists(DATASET_DIR):
        try:
            folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
            if "RESTING" in folders:
                vocab_list.append("RESTING")
                folders.remove("RESTING")
            else:
                vocab_list.append("RESTING")

            for f in sorted(folders):
                display_name = folder_to_display_phrase(f)
                if display_name not in vocab_list:
                    vocab_list.append(display_name)
        except Exception:
            pass

    if not vocab_list:
        if os.path.exists(VOCAB_FILE):
            try:
                with open(VOCAB_FILE, "r", encoding="utf-8") as f:
                    v = json.load(f)
                    if isinstance(v, list) and len(v) > 0:
                        vocab_list = v
            except Exception:
                pass

    if not vocab_list:
        vocab_list = list(DEFAULT_VOCABULARY)

    # Save cached vocab.json if writable
    try:
        with open(VOCAB_FILE, "w", encoding="utf-8") as f:
            json.dump(vocab_list, f, indent=2)
    except Exception:
        pass

    return vocab_list


def synthesize_custom_trajectory(phrase: str, seq_len: int = 30) -> np.ndarray:
    """
    Synthesizes an articulatory phonetic multi-syllable trajectory for any custom phrase or sentence.
    Distributes aperture peaks proportionally across words.
    """
    words = phrase.strip().split()
    n_words = max(1, len(words))
    t = np.linspace(0, 1, seq_len)
    
    h = np.ones(seq_len) * 0.05
    w = np.ones(seq_len) * 0.45
    z = np.zeros(seq_len)

    for i in range(n_words):
        t_center = (i + 1) / (n_words + 1)
        sigma = 0.08 / np.sqrt(n_words)
        h += 0.28 * np.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
        w += 0.06 * np.sin(2 * np.pi * t)

    aspect = h / (w + 1e-5)
    outer_h = h * 1.5 + 0.08
    upper_thick = 0.04 + 0.02 * h
    lower_thick = 0.05 + 0.03 * h
    corner_angle = 0.02 * np.sin(np.linspace(0, np.pi, seq_len))
    area = np.pi * (h / 2.0) * (w / 2.0)
    
    dh = np.gradient(h) * 5.0
    dw = np.gradient(w) * 5.0
    d2h = np.gradient(dh) * 2.0
    
    features = np.stack([
        h, w, aspect, outer_h, upper_thick, lower_thick,
        corner_angle, area, dh, dw, d2h, z
    ], axis=-1)
    return features.astype(np.float32)


def add_custom_phrase(phrase: str) -> str:
    """
    Creates a new dedicated folder for ANY custom word or full sentence.
    Leaves the folder completely empty (0 files) so the user can record their 10 samples.
    """
    clean_name = clean_folder_name(phrase)
    if not clean_name:
        raise ValueError("Invalid phrase entered")

    folder_path = os.path.join(DATASET_DIR, clean_name)
    try:
        os.makedirs(folder_path, exist_ok=True)
    except (OSError, PermissionError):
        pass

    get_dynamic_vocabulary()  # Refresh vocab.json
    return folder_to_display_phrase(clean_name)


def save_user_sample(phrase: str, tensor_data: List[List[float]], sample_num: Optional[int] = None) -> str:
    """
    Saves a captured 30x12 kinematic trajectory into dataset/<PHRASE>/sample_<num>_<timestamp>.npy
    """
    folder_name = clean_folder_name(phrase)
    arr = np.array(tensor_data, dtype=np.float32)
    if arr.shape != (30, 12):
        raise ValueError(f"Expected shape (30, 12), got {arr.shape}")

    timestamp = int(time.time() * 1000)
    tag = f"sample_{sample_num:02d}_{timestamp}.npy" if sample_num else f"sample_{timestamp}.npy"

    try:
        folder_path = os.path.join(DATASET_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, tag)
        np.save(file_path, arr)
        return file_path
    except (OSError, PermissionError):
        tmp_folder = os.path.join("/tmp", "dataset", folder_name)
        os.makedirs(tmp_folder, exist_ok=True)
        file_path = os.path.join(tmp_folder, tag)
        np.save(file_path, arr)
        return file_path


def clear_phrase_samples(phrase: str) -> int:
    """Deletes all recording files for a specific phrase."""
    folder_name = clean_folder_name(phrase)
    folder_path = os.path.join(DATASET_DIR, folder_name)
    deleted = 0
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if f.endswith(".npy"):
                try:
                    os.remove(os.path.join(folder_path, f))
                    deleted += 1
                except Exception:
                    pass
    return deleted


def reset_entire_dataset() -> int:
    """Empties all recorded .npy files from every folder in dataset/."""
    deleted = 0
    if os.path.exists(DATASET_DIR):
        for root, dirs, files in os.walk(DATASET_DIR):
            for f in files:
                if f.endswith(".npy"):
                    try:
                        os.remove(os.path.join(root, f))
                        deleted += 1
                    except Exception:
                        pass
    return deleted


def get_dataset_statistics() -> Dict:
    vocab = get_dynamic_vocabulary()
    per_word_counts = {}
    real_recordings = {}
    total_samples = 0

    for phrase in vocab:
        folder_name = clean_folder_name(phrase)
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
            count = len(files)
        else:
            count = 0
        per_word_counts[phrase] = count
        real_recordings[phrase] = count
        total_samples += count

    return {
        "vocabulary": vocab,
        "per_word_counts": per_word_counts,
        "real_recordings": real_recordings,
        "total_samples": total_samples,
        "vocabulary_size": len(vocab)
    }


class UserKinematicDataset(Dataset):
    """
    Loads all recorded .npy files from folders in dataset/ with biological augmentation.
    Filters to only words that have at least 1 recorded sample.
    """
    def __init__(self, augment_factor: int = 8, active_words: Optional[List[str]] = None):
        self.samples = []
        self.labels = []

        all_vocab = get_dynamic_vocabulary()
        if active_words is not None:
            vocab = [w for w in active_words if w in all_vocab]
        else:
            # Only include words that actually have at least 1 sample recorded!
            vocab = []
            for phrase in all_vocab:
                folder_name = clean_folder_name(phrase)
                folder_path = os.path.join(DATASET_DIR, folder_name)
                if os.path.exists(folder_path):
                    f_list = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
                    if len(f_list) > 0:
                        vocab.append(phrase)

        self.vocabulary = vocab
        vocab_map = {word: idx for idx, word in enumerate(vocab)}

        for phrase in vocab:
            label_idx = vocab_map[phrase]
            folder_name = clean_folder_name(phrase)
            folder_path = os.path.join(DATASET_DIR, folder_name)

            if not os.path.exists(folder_path):
                continue

            file_names = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

            for fname in file_names:
                fpath = os.path.join(folder_path, fname)
                try:
                    raw_arr = np.load(fpath).astype(np.float32)
                    if raw_arr.shape == (30, 12):
                        self.samples.append(raw_arr)
                        self.labels.append(label_idx)

                        # Biological Augmentation (Motor Speed Variation + Spatial Scale + Sensor Noise)
                        for _ in range(augment_factor):
                            aug = raw_arr.copy()
                            shift = np.random.randint(-2, 3)
                            aug = np.roll(aug, shift, axis=0)
                            scale = np.random.uniform(0.92, 1.08)
                            aug[:, :6] *= scale
                            noise = np.random.normal(0, 0.005, aug.shape)
                            aug += noise

                            self.samples.append(aug.astype(np.float32))
                            self.labels.append(label_idx)
                except Exception as e:
                    print(f"Error loading {fpath}: {e}")

        self.samples = np.array(self.samples, dtype=np.float32) if len(self.samples) > 0 else np.zeros((0, 30, 12), dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64) if len(self.labels) > 0 else np.zeros((0,), dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if HAS_TORCH and torch is not None:
            return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])
        return self.samples[idx], self.labels[idx]
