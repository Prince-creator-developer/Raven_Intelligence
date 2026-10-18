import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from lip_model import VisemeTemporalNet, VOCABULARY, VOCAB_MAP

# Kinematic Trajectory Generators for English Visemes
def generate_base_kinematic_trajectory(word_idx: int, seq_len: int = 30) -> np.ndarray:
    """
    Generates anatomically grounded lip aperture trajectories (height, width)
    based on articulatory phonetics for each vocabulary word.
    """
    t = np.linspace(0, 1, seq_len)
    h = np.ones(seq_len) * 0.06  # Base resting height
    w = np.ones(seq_len) * 0.45  # Base resting width
    z = np.zeros(seq_len)        # Protrusion

    if word_idx == 0:  # RESTING
        h += 0.01 * np.sin(2 * np.pi * t)
        w += 0.01 * np.cos(2 * np.pi * t)

    elif word_idx == 1:  # HELP: Open /ɛ/ -> Lateral /l/ -> Bilabial seal /p/
        # Opens at t=0.3, seals shut at t=0.8
        h = 0.05 + 0.32 * np.exp(-((t - 0.35) ** 2) / 0.02)
        h[t > 0.75] = 0.01  # Bilabial /p/ seal
        w = 0.45 + 0.08 * np.exp(-((t - 0.35) ** 2) / 0.04)

    elif word_idx == 2:  # WATER: Rounded /w/ -> Open /ɔ/ -> Flap /t/ -> Neutral /ɚ/
        # Initial lip rounding (narrow width)
        w = 0.45 - 0.20 * np.exp(-((t - 0.20) ** 2) / 0.03) + 0.06 * np.exp(-((t - 0.6) ** 2) / 0.05)
        h = 0.06 + 0.35 * np.exp(-((t - 0.50) ** 2) / 0.04)
        z = 0.15 * np.exp(-((t - 0.20) ** 2) / 0.03)  # Protrusion during /w/

    elif word_idx == 3:  # YES: Palatal spread /j/ -> Mid-open /ɛ/ -> Fricative /s/
        w = 0.45 + 0.18 * np.exp(-((t - 0.30) ** 2) / 0.05) + 0.12 * np.exp(-((t - 0.75) ** 2) / 0.04)
        h = 0.05 + 0.28 * np.exp(-((t - 0.45) ** 2) / 0.03)

    elif word_idx == 4:  # NO: Alveolar /n/ -> Progressive rounding /oʊ/
        h = 0.05 + 0.30 * np.exp(-((t - 0.40) ** 2) / 0.04)
        # Lip constriction towards end
        w = 0.45 - 0.18 * (t ** 1.5)
        z = 0.12 * (t ** 2)

    elif word_idx == 5:  # DOCTOR: Alveolar /d/ -> Large open /ɑ/ -> Stop /k/ -> /ɚ/
        h = 0.05 + 0.44 * np.exp(-((t - 0.35) ** 2) / 0.03) + 0.15 * np.exp(-((t - 0.75) ** 2) / 0.04)
        w = 0.45 + 0.05 * np.sin(np.pi * t)

    elif word_idx == 6:  # PAIN: Bilabial seal /p/ -> Diphthong /eɪ/ -> Nasal /n/
        h = np.zeros(seq_len)
        h[t < 0.25] = 0.008  # Initial tight seal
        h[t >= 0.25] = 0.05 + 0.26 * np.exp(-((t[t >= 0.25] - 0.50) ** 2) / 0.03)
        w = 0.45 + 0.14 * np.exp(-((t - 0.55) ** 2) / 0.05)

    elif word_idx == 7:  # THANK YOU: Dental /θ/ -> Open /æ/ -> Rounded /u/
        h = 0.08 + 0.36 * np.exp(-((t - 0.35) ** 2) / 0.03) + 0.12 * np.exp(-((t - 0.75) ** 2) / 0.03)
        w = 0.45 + 0.08 * np.exp(-((t - 0.35) ** 2) / 0.04) - 0.16 * np.exp(-((t - 0.80) ** 2) / 0.03)

    elif word_idx == 8:  # TIRED: Stop /t/ -> Wide /aɪ/ -> Stop /d/
        h = 0.05 + 0.42 * np.exp(-((t - 0.40) ** 2) / 0.04) - 0.20 * np.exp(-((t - 0.70) ** 2) / 0.02)
        w = 0.45 + 0.12 * np.exp(-((t - 0.45) ** 2) / 0.06)

    elif word_idx == 9:  # NURSE: Alveolar /n/ -> Central /ɚ/ -> Fricative /s/
        h = 0.05 + 0.22 * np.exp(-((t - 0.45) ** 2) / 0.05)
        w = 0.45 - 0.06 * np.exp(-((t - 0.45) ** 2) / 0.04) + 0.10 * np.exp(-((t - 0.80) ** 2) / 0.03)

    elif word_idx == 10: # BREATHE: Seal /b/ -> Spread /i/ -> Dental /ð/
        h = 0.01 * (t < 0.2) + (0.05 + 0.18 * np.exp(-((t - 0.50) ** 2) / 0.04)) * (t >= 0.2)
        w = 0.45 + 0.16 * np.exp(-((t - 0.55) ** 2) / 0.05)

    elif word_idx == 11: # MEDICINE: Multi-syllable envelope
        h = 0.01 * (t < 0.15) + 0.25 * np.exp(-((t - 0.35) ** 2) / 0.02) + 0.20 * np.exp(-((t - 0.65) ** 2) / 0.02)
        w = 0.45 + 0.08 * np.sin(2 * np.pi * t)

    return h, w, z


def construct_12d_feature_matrix(h: np.ndarray, w: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Constructs the 12-dimensional kinematic vector across 30 temporal frames:
    1: vertical_aperture
    2: horizontal_aperture
    3: aspect_ratio
    4: outer_height
    5: upper_lip_thickness
    6: lower_lip_thickness
    7: mouth_corner_angle
    8: lip_area_ratio
    9: delta_height (velocity)
    10: delta_width (velocity)
    11: accel_height
    12: protrusion_depth
    """
    seq_len = len(h)
    aspect = h / (w + 1e-5)
    outer_h = h * 1.5 + 0.08
    upper_thick = 0.04 + 0.02 * h
    lower_thick = 0.05 + 0.03 * h
    corner_angle = 0.02 * np.sin(np.linspace(0, np.pi, seq_len))
    area = np.pi * (h / 2.0) * (w / 2.0)
    
    # Kinematic 1st and 2nd derivatives
    dh = np.gradient(h)
    dw = np.gradient(w)
    d2h = np.gradient(dh)
    
    features = np.stack([
        h, w, aspect, outer_h, upper_thick, lower_thick,
        corner_angle, area, dh * 5.0, dw * 5.0, d2h * 10.0, z
    ], axis=-1)
    
    return features.astype(np.float32)


class SyntheticVisemeDataset(Dataset):
    def __init__(self, samples_per_class: int = 400):
        self.data = []
        self.labels = []
        np.random.seed(42)

        for word_idx in range(len(VOCABULARY)):
            base_h, base_w, base_z = generate_base_kinematic_trajectory(word_idx)

            for _ in range(samples_per_class):
                # 1. Biological Amplitude Variation (Speaker face size / articulatory range)
                h_scale = np.random.uniform(0.85, 1.15)
                w_scale = np.random.uniform(0.90, 1.10)
                
                # 2. Articulatory Speed Time-Warping (+/- 20%)
                shift = np.random.randint(-3, 4)
                h_aug = np.roll(base_h * h_scale, shift)
                w_aug = np.roll(base_w * w_scale, shift)
                z_aug = np.roll(base_z, shift)

                # 3. Physiological Micro-Tremor Noise
                h_aug += np.random.normal(0, 0.008, 30)
                w_aug += np.random.normal(0, 0.010, 30)
                z_aug += np.random.normal(0, 0.005, 30)

                # Keep bounds realistic
                h_aug = np.clip(h_aug, 0.005, 0.65)
                w_aug = np.clip(w_aug, 0.20, 0.75)

                feat_matrix = construct_12d_feature_matrix(h_aug, w_aug, z_aug)
                self.data.append(feat_matrix)
                self.labels.append(word_idx)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])


def train_lip_model():
    print("=" * 65)
    print("  [MODEL TRAINING] SILENTVOICE AI - BIOMECHANICAL KINEMATIC NET")
    print("  Vocabulary Size:", len(VOCABULARY))
    print("  Classes:", ", ".join(VOCABULARY))
    print("=" * 65)

    # Generate dataset
    print("[1/4] Generating Biomechanical Viseme Kinematics Dataset...")
    dataset = SyntheticVisemeDataset(samples_per_class=450)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    print(f"      Total Samples: {len(dataset)} | Train: {train_size} | Val: {val_size}")

    # Initialize model
    print("[2/4] Initializing VisemeTemporalNet (BiLSTM + Temporal Attention)...")
    device = torch.device("cpu")
    model = VisemeTemporalNet(input_dim=12, hidden_dim=64, num_classes=len(VOCABULARY)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    epochs = 35
    best_acc = 0.0
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lip_model.pth")

    print("[3/4] Starting Rigorous Neural Training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            probs, _ = model(x_batch)
            loss = criterion(probs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(probs, dim=-1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                probs, _ = model(x_val)
                preds = torch.argmax(probs, dim=-1)
                val_correct += (preds == y_val).sum().item()
                val_total += len(y_val)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% (Best: {best_acc*100:.1f}%)")

    elapsed = time.time() - start_time
    print(f"\n[4/4] Training Complete in {elapsed:.1f}s!")
    print(f"      Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"      Saved Model Weights to: {save_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    train_lip_model()
