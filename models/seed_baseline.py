import os
import numpy as np
from dataset_manager import DATASET_DIR, VOCABULARY, clean_word_folder_name
from train_model import generate_base_kinematic_trajectory, construct_12d_feature_matrix

def seed_baseline_dataset(samples_per_word: int = 15):
    print("Seeding initial baseline kinematic files into word folders...")
    np.random.seed(1337)

    for word_idx, word in enumerate(VOCABULARY):
        folder = os.path.join(DATASET_DIR, clean_word_folder_name(word))
        os.makedirs(folder, exist_ok=True)
        base_h, base_w, base_z = generate_base_kinematic_trajectory(word_idx)

        for i in range(samples_per_word):
            h_scale = np.random.uniform(0.90, 1.10)
            w_scale = np.random.uniform(0.92, 1.08)
            shift = np.random.randint(-2, 3)

            h_aug = np.roll(base_h * h_scale, shift)
            w_aug = np.roll(base_w * w_scale, shift)
            z_aug = np.roll(base_z, shift)

            h_aug += np.random.normal(0, 0.005, 30)
            w_aug += np.random.normal(0, 0.006, 30)
            z_aug += np.random.normal(0, 0.003, 30)

            h_aug = np.clip(h_aug, 0.01, 0.60)
            w_aug = np.clip(w_aug, 0.22, 0.70)

            features = construct_12d_feature_matrix(h_aug, w_aug, z_aug)
            file_path = os.path.join(folder, f"baseline_{i+1:03d}.npy")
            np.save(file_path, features)

    print(f"Successfully seeded {samples_per_word} files in each of the {len(VOCABULARY)} word folders.")

if __name__ == "__main__":
    seed_baseline_dataset(samples_per_word=15)
