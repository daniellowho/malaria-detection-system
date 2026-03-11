"""
=============================================================
 Malaria Detection – Model 1: Custom CNN
=============================================================
Architecture:
  Block 1: Conv(32)  → BN → ReLU → MaxPool → Dropout
  Block 2: Conv(64)  → BN → ReLU → MaxPool → Dropout
  Block 3: Conv(128) → BN → ReLU → MaxPool → Dropout
  Block 4: Conv(256) → BN → ReLU → GAP
  Head   : Dense(256) → Dropout → Dense(1, sigmoid)
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ─── Hyper-parameters ─────────────────────────────────────
IMG_SIZE    = (128, 128)
IMG_SHAPE   = (128, 128, 3)
EPOCHS      = 50
BATCH_SIZE  = 32
LR_INITIAL  = 1e-3
L2_REG      = 1e-4
DROPOUT     = 0.4
MODEL_NAME  = "custom_cnn"
SAVE_PATH   = f"saved_models/{MODEL_NAME}"


# ─── 1. Build model ───────────────────────────────────────
def build_custom_cnn(input_shape: tuple = IMG_SHAPE,
                     dropout: float = DROPOUT,
                     l2: float = L2_REG) -> keras.Model:

    reg = regularizers.l2(l2)

    inputs = keras.Input(shape=input_shape, name="cell_image")
    x = inputs

    # ── Convolutional blocks ──────────────────────────────
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if filters < 256:
            x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

    # ── Global Average Pooling ────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)

    # ── Classification head ───────────────────────────────
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs, outputs, name=MODEL_NAME)
    return model


# ─── 2. Compile helper ────────────────────────────────────
def compile_model(model: keras.Model, lr: float = LR_INITIAL):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ─── 3. Callbacks ─────────────────────────────────────────
def get_callbacks(model_name: str = MODEL_NAME):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs",         exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=f"saved_models/{model_name}_best.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=f"logs/{model_name}",
            histogram_freq=1,
        ),
        keras.callbacks.CSVLogger(
            f"logs/{model_name}_history.csv"
        ),
    ]


# ─── 4. Train ─────────────────────────────────────────────
def train(train_gen, val_gen, epochs: int = EPOCHS):
    model = build_custom_cnn()
    model = compile_model(model)
    model.summary()

    print(f"\n🚀  Training {MODEL_NAME} for {epochs} epochs …\n")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1,
    )
    model.save(f"{SAVE_PATH}.keras")
    print(f"\n✅  Model saved → {SAVE_PATH}.keras")
    return model, history


# ─── 5. Plot training curves ──────────────────────────────
def plot_history(history, title: str = MODEL_NAME):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title} – Training History", fontsize=14, fontweight="bold")

    pairs = [
        ("accuracy",  "val_accuracy",  "Accuracy"),
        ("loss",      "val_loss",      "Loss"),
        ("auc",       "val_auc",       "AUC-ROC"),
    ]
    for ax, (tr, vl, name) in zip(axes, pairs):
        ax.plot(history.history[tr], label="Train",      color="#2196F3")
        ax.plot(history.history[vl], label="Validation", color="#F44336")
        ax.set_title(name); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title}_history.png", dpi=150)
    plt.show()


# ─── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke-test: build and print summary without real data
    m = build_custom_cnn()
    compile_model(m)
    m.summary()
    tf.keras.utils.plot_model(m, to_file="plots/custom_cnn_architecture.png",
                               show_shapes=True, dpi=96)
    print("\n✅  Custom CNN ready for training.")
