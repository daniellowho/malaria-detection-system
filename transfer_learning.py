"""
=============================================================
 Malaria Detection – Transfer Learning Models
 ▸ MobileNetV2  (lightweight, mobile-ready)
 ▸ EfficientNetB0 (state-of-the-art accuracy/efficiency)
=============================================================
Strategy
--------
Phase 1 – Feature extraction  (base frozen, train head only)
Phase 2 – Fine-tuning          (unfreeze top layers, low LR)
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─── Shared config ────────────────────────────────────────
IMG_SHAPE   = (128, 128, 3)
EPOCHS_P1   = 20          # Phase 1: frozen base
EPOCHS_P2   = 30          # Phase 2: fine-tune
LR_P1       = 1e-3
LR_P2       = 1e-5
DROPOUT     = 0.4
UNFREEZE_AT = 100         # unfreeze layers after index for fine-tuning


# ─── Generic classification head ──────────────────────────
def _classification_head(base_output, dropout: float, name: str):
    x = layers.GlobalAveragePooling2D()(base_output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64,  activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    return out


def _compile(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def _callbacks(name):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs(f"logs/{name}", exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            f"saved_models/{name}_best.keras",
            monitor="val_auc", mode="max",
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=8,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-7, verbose=1),
        keras.callbacks.TensorBoard(log_dir=f"logs/{name}"),
        keras.callbacks.CSVLogger(f"logs/{name}_history.csv"),
    ]


# ═══════════════════════════════════════════════════════════
#  MobileNetV2
# ═══════════════════════════════════════════════════════════
class MobileNetV2Detector:
    MODEL_NAME = "mobilenetv2"

    def __init__(self, input_shape=IMG_SHAPE, dropout=DROPOUT):
        self.input_shape = input_shape
        self.dropout     = dropout
        self.model       = None
        self.history_p1  = None
        self.history_p2  = None

    def build(self):
        base = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False          # freeze for Phase 1

        inputs  = keras.Input(shape=self.input_shape)
        x       = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x       = base(x, training=False)
        outputs = _classification_head(x, self.dropout, self.MODEL_NAME)

        self.model = keras.Model(inputs, outputs, name=self.MODEL_NAME)
        self._base  = base
        return self.model

    def train_phase1(self, train_gen, val_gen, epochs=EPOCHS_P1):
        """Phase 1 – train head with frozen base."""
        print(f"\n📌  MobileNetV2 Phase 1 – feature extraction ({epochs} epochs)")
        _compile(self.model, LR_P1)
        self.history_p1 = self.model.fit(
            train_gen, epochs=epochs,
            validation_data=val_gen,
            callbacks=_callbacks(f"{self.MODEL_NAME}_p1"),
        )
        return self.history_p1

    def train_phase2(self, train_gen, val_gen,
                     unfreeze_at=UNFREEZE_AT, epochs=EPOCHS_P2):
        """Phase 2 – unfreeze top layers and fine-tune."""
        print(f"\n🔓  MobileNetV2 Phase 2 – fine-tuning top layers ({epochs} epochs)")
        self._base.trainable = True
        for layer in self._base.layers[:unfreeze_at]:
            layer.trainable = False

        _compile(self.model, LR_P2)
        self.history_p2 = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=_callbacks(f"{self.MODEL_NAME}_p2"),
        )
        self.model.save(f"saved_models/{self.MODEL_NAME}.keras")
        print(f"✅  MobileNetV2 saved → saved_models/{self.MODEL_NAME}.keras")
        return self.history_p2


# ═══════════════════════════════════════════════════════════
#  EfficientNetB0
# ═══════════════════════════════════════════════════════════
class EfficientNetB0Detector:
    MODEL_NAME = "efficientnetb0"

    def __init__(self, input_shape=IMG_SHAPE, dropout=DROPOUT):
        self.input_shape = input_shape
        self.dropout     = dropout
        self.model       = None
        self.history_p1  = None
        self.history_p2  = None

    def build(self):
        base = keras.applications.EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False

        inputs  = keras.Input(shape=self.input_shape)
        # EfficientNet has its own internal preprocessing
        x       = base(inputs, training=False)
        # Squeeze-and-Excitation-style feature attention
        x       = layers.GlobalAveragePooling2D()(x)
        x       = layers.Dense(256, activation="swish")(x)
        x       = layers.BatchNormalization()(x)
        x       = layers.Dropout(self.dropout)(x)
        x       = layers.Dense(64, activation="swish")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

        self.model = keras.Model(inputs, outputs, name=self.MODEL_NAME)
        self._base  = base
        return self.model

    def train_phase1(self, train_gen, val_gen, epochs=EPOCHS_P1):
        print(f"\n📌  EfficientNetB0 Phase 1 – feature extraction ({epochs} epochs)")
        _compile(self.model, LR_P1)
        self.history_p1 = self.model.fit(
            train_gen, epochs=epochs,
            validation_data=val_gen,
            callbacks=_callbacks(f"{self.MODEL_NAME}_p1"),
        )
        return self.history_p1

    def train_phase2(self, train_gen, val_gen,
                     unfreeze_at=UNFREEZE_AT, epochs=EPOCHS_P2):
        print(f"\n🔓  EfficientNetB0 Phase 2 – fine-tuning ({epochs} epochs)")
        self._base.trainable = True
        for layer in self._base.layers[:unfreeze_at]:
            layer.trainable = False

        _compile(self.model, LR_P2)
        self.history_p2 = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=_callbacks(f"{self.MODEL_NAME}_p2"),
        )
        self.model.save(f"saved_models/{self.MODEL_NAME}.keras")
        print(f"✅  EfficientNetB0 saved → saved_models/{self.MODEL_NAME}.keras")
        return self.history_p2


# ─── Combined training runner ─────────────────────────────
def train_all_transfer_models(train_gen, val_gen):
    results = {}

    for DetectorClass in [MobileNetV2Detector, EfficientNetB0Detector]:
        det = DetectorClass()
        det.build()
        det.model.summary()
        det.train_phase1(train_gen, val_gen)
        det.train_phase2(train_gen, val_gen)
        results[det.MODEL_NAME] = det

    return results


# ─── Training curve plotter ───────────────────────────────
def plot_combined_history(results: dict, metric="val_auc"):
    plt.figure(figsize=(10, 5))
    colors = {"mobilenetv2": "#2196F3", "efficientnetb0": "#4CAF50",
              "custom_cnn": "#FF9800"}

    for name, det in results.items():
        hist = det.history_p2 or det.history_p1
        if hist and metric in hist.history:
            plt.plot(hist.history[metric],
                     label=name, color=colors.get(name, "#888"))

    plt.title(f"Validation {metric.replace('val_','').upper()} – All Models")
    plt.xlabel("Epoch"); plt.ylabel(metric)
    plt.legend(); plt.grid(alpha=0.3)
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/combined_{metric}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Smoke-test architecture builds
    for Cls in [MobileNetV2Detector, EfficientNetB0Detector]:
        d = Cls(); d.build()
        print(f"\n{Cls.__name__}")
        d.model.summary(line_length=90)
    print("\n✅  Transfer-learning models ready.")
