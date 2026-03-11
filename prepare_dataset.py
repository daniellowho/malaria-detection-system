"""
=============================================================
 Malaria Detection - Dataset Preparation & Preprocessing
=============================================================
Dataset: NIH Cell Images for Detecting Malaria (Kaggle)
URL: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
=============================================================
"""

import os
import shutil
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─── Config ───────────────────────────────────────────────
DATASET_ROOT   = "cell_images"          # raw NIH dataset root
OUTPUT_ROOT    = "data"                  # organised split root
IMG_SIZE       = (128, 128)
BATCH_SIZE     = 32
VAL_SPLIT      = 0.20
RANDOM_SEED    = 42
CLASSES        = ["Parasitized", "Uninfected"]


# ─── 1. Organise raw folder into train / val ───────────────
def split_dataset(dataset_root: str, output_root: str,
                  val_split: float = VAL_SPLIT,
                  seed: int = RANDOM_SEED):
    """
    Expected raw layout
    -------------------
    cell_images/
        Parasitized/  *.png
        Uninfected/   *.png

    Output layout
    -------------
    data/
        train/
            Parasitized/
            Uninfected/
        val/
            Parasitized/
            Uninfected/
    """
    random.seed(seed)

    for split in ["train", "val"]:
        for cls in CLASSES:
            Path(f"{output_root}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

    stats = {}
    for cls in CLASSES:
        src_dir = Path(dataset_root) / cls
        images  = [f for f in src_dir.iterdir()
                   if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}]
        random.shuffle(images)

        n_val   = int(len(images) * val_split)
        val_set = images[:n_val]
        trn_set = images[n_val:]

        for img in trn_set:
            shutil.copy(img, f"{output_root}/train/{cls}/{img.name}")
        for img in val_set:
            shutil.copy(img, f"{output_root}/val/{cls}/{img.name}")

        stats[cls] = {"train": len(trn_set), "val": len(val_set)}
        print(f"  {cls:>14} → train: {len(trn_set):,}  val: {len(val_set):,}")

    print(f"\n✅  Dataset split complete → '{output_root}/'")
    return stats


# ─── 2. Data Generators ───────────────────────────────────
def build_generators(data_root: str = "data",
                     img_size: tuple = IMG_SIZE,
                     batch_size: int = BATCH_SIZE):
    """
    Returns (train_gen, val_gen) with augmentation on training set.
    """

    # Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.20,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    # Validation: only normalise
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        f"{data_root}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASSES,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        f"{data_root}/val",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASSES,
        shuffle=False,
    )

    print(f"\nClass indices: {train_gen.class_indices}")
    print(f"  Train batches : {len(train_gen)}")
    print(f"  Val   batches : {len(val_gen)}")
    return train_gen, val_gen


# ─── 3. Exploratory visualisation ─────────────────────────
def visualise_samples(data_root: str = "data", n: int = 10):
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 6))
    fig.suptitle("Sample Cell Images", fontsize=16, fontweight="bold", y=1.02)

    for col, cls in enumerate(CLASSES):
        folder = Path(data_root) / "train" / cls
        imgs   = random.sample(list(folder.iterdir()), n)
        for row, img_path in enumerate(imgs):
            ax = axes[col, row]
            ax.imshow(Image.open(img_path))
            ax.axis("off")
            if row == 0:
                ax.set_title(cls, fontsize=12, color="red" if cls == "Parasitized" else "green")

    plt.tight_layout()
    plt.savefig("sample_cells.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📷  sample_cells.png saved")


def plot_class_distribution(stats: dict):
    labels  = list(stats.keys())
    t_vals  = [stats[c]["train"] for c in labels]
    v_vals  = [stats[c]["val"]   for c in labels]

    x  = np.arange(len(labels))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, t_vals, w, label="Train",      color="#2196F3")
    b2 = ax.bar(x + w/2, v_vals, w, label="Validation", color="#FF9800")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Image Count"); ax.set_title("Class Distribution")
    ax.legend(); ax.bar_label(b1, padding=3); ax.bar_label(b2, padding=3)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()


# ─── 4. tf.data pipeline (optional high-perf alternative) ─
def build_tf_datasets(data_root: str = "data",
                      img_size: tuple = IMG_SIZE,
                      batch_size: int = BATCH_SIZE):
    """
    High-performance tf.data pipeline with prefetching.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, label

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_root}/train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED,
    ).map(preprocess, num_parallel_calls=AUTOTUNE) \
     .map(augment,    num_parallel_calls=AUTOTUNE) \
     .prefetch(AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_root}/val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    ).map(preprocess, num_parallel_calls=AUTOTUNE) \
     .prefetch(AUTOTUNE)

    return train_ds, val_ds


# ─── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(" Malaria Detection – Dataset Preparation")
    print("=" * 55)

    print("\n[1/3] Splitting dataset …")
    stats = split_dataset(DATASET_ROOT, OUTPUT_ROOT)

    print("\n[2/3] Building data generators …")
    train_gen, val_gen = build_generators()

    print("\n[3/3] Visualising samples …")
    plot_class_distribution(stats)
    visualise_samples()

    print("\n✅  All preprocessing steps complete!")
