import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from lip_model import VisemeTemporalNet
from dataset_manager import (
    UserKinematicDataset,
    get_dataset_statistics,
    get_dynamic_vocabulary,
    VOCAB_FILE
)

def train_custom_100_epochs(epochs: int = 100, batch_size: int = 32, lr: float = 1e-3) -> dict:
    """
    Trains VisemeTemporalNet on all folders stored in dataset/ for 100 epochs.
    Supports arbitrary custom words, medical terms, and full sentences.
    """
    print("=" * 65)
    print(f"  [TRAINING ENGINE] RUNNING {epochs} EPOCHS ON LOCAL DATASET")
    print("=" * 65)

    vocab = get_dynamic_vocabulary()
    num_classes = len(vocab)
    print(f"Dynamic Vocabulary Size: {num_classes} classes")
    print(f"Active Classes: {', '.join(vocab)}")

    stats = get_dataset_statistics()
    total_raw_files = stats["total_samples"]
    print(f"Loaded {total_raw_files} recorded kinematic files across {num_classes} directories.")

    # Load dataset with augmentation
    dataset = UserKinematicDataset(augment_factor=6)
    total_samples = len(dataset)
    print(f"Total augmented training tensors: {total_samples}")

    if total_samples < 20:
        raise ValueError(f"Insufficient samples to train. Found only {total_samples} samples. Record more samples first!")

    train_size = int(0.85 * total_samples)
    val_size = total_samples - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = VisemeTemporalNet(input_dim=12, hidden_dim=64, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lip_model.pth")
    history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            probs, _ = model(x_b)
            loss = criterion(probs, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_b)
            preds = torch.argmax(probs, dim=-1)
            train_correct += (preds == y_b).sum().item()
            train_total += len(y_b)

        scheduler.step()
        train_acc = train_correct / max(train_total, 1)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                probs, _ = model(x_v)
                preds = torch.argmax(probs, dim=-1)
                val_correct += (preds == y_v).sum().item()
                val_total += len(y_v)

        val_acc = val_correct / max(val_total, 1)
        history.append({
            "epoch": epoch,
            "loss": round(train_loss / max(train_total, 1), 4),
            "train_acc": round(train_acc * 100, 1),
            "val_acc": round(val_acc * 100, 1)
        })

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:03d}/{epochs:03d} | Loss: {train_loss/max(train_total,1):.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% (Best: {best_val_acc*100:.1f}%)")

    total_time = round(time.time() - t_start, 2)
    print("=" * 65)
    print(f"  Training Complete in {total_time}s! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"  Model saved to: {save_path}")
    print("=" * 65)

    return {
        "status": "success",
        "vocabulary": vocab,
        "epochs_trained": epochs,
        "best_val_acc": round(best_val_acc * 100, 2),
        "final_loss": history[-1]["loss"] if history else 0.0,
        "training_time_s": total_time,
        "total_samples": total_samples,
        "raw_files": total_raw_files,
        "save_path": save_path
    }

if __name__ == "__main__":
    train_custom_100_epochs(epochs=100)
