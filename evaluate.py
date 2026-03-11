"""
=============================================================
 Malaria Detection – Comprehensive Model Evaluation
=============================================================
Metrics:  Accuracy · Precision · Recall · F1
          ROC-AUC · PR-AUC · Confusion Matrix
          Class Activation Maps (Grad-CAM)
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, matthews_corrcoef,
)
from pathlib import Path


CLASSES    = ["Uninfected", "Parasitized"]   # 0 = Uninfected, 1 = Parasitized
IMG_SIZE   = (128, 128)
THRESHOLD  = 0.5
PLOTS_DIR  = "plots/evaluation"


os.makedirs(PLOTS_DIR, exist_ok=True)


# ─── 1. Collect predictions ───────────────────────────────
def get_predictions(model, val_gen, threshold: float = THRESHOLD):
    """Return y_true, y_pred_proba, y_pred_label."""
    val_gen.reset()
    y_proba = model.predict(val_gen, verbose=1).ravel()
    y_true  = val_gen.classes
    y_pred  = (y_proba >= threshold).astype(int)
    return y_true, y_proba, y_pred


# ─── 2. Core metrics ──────────────────────────────────────
def compute_metrics(y_true, y_proba, y_pred, model_name="model"):
    roc_auc = auc(*roc_curve(y_true, y_proba)[:2])
    pr_auc  = average_precision_score(y_true, y_proba)
    f1      = f1_score(y_true, y_pred)
    mcc     = matthews_corrcoef(y_true, y_pred)

    report  = classification_report(y_true, y_pred,
                                     target_names=CLASSES, output_dict=True)
    metrics = {
        "model":     model_name,
        "accuracy":  report["accuracy"],
        "precision": report["Parasitized"]["precision"],
        "recall":    report["Parasitized"]["recall"],
        "f1":        f1,
        "roc_auc":   roc_auc,
        "pr_auc":    pr_auc,
        "mcc":       mcc,
    }

    print(f"\n{'='*55}")
    print(f"  {model_name.upper()} Evaluation Results")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<14}: {v:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    return metrics


# ─── 3. Confusion Matrix ──────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name="model"):
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # normalised

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrix – {model_name}", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_n],
        ["d", ".2%"],
        ["Raw Counts", "Normalised"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="RdYlGn",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"📊  {path} saved")

    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Sensitivity (Recall) : {tp/(tp+fn):.4f}")
    print(f"  Specificity          : {tn/(tn+fp):.4f}")


# ─── 4. ROC & PR curves ───────────────────────────────────
def plot_roc_pr(results: dict):
    """
    results: {model_name: (y_true, y_proba)}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for (name, (yt, yp)), color in zip(results.items(), colors):
        # ROC
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc     = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, lw=2,
                 label=f"{name}  AUC={roc_auc:.4f}")

        # PR
        pre, rec, _ = precision_recall_curve(yt, yp)
        pr_auc      = average_precision_score(yt, yp)
        ax2.plot(rec, pre, color=color, lw=2,
                 label=f"{name}  AP={pr_auc:.4f}")

    # ROC decoration
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set(title="ROC Curves", xlabel="FPR", ylabel="TPR",
            xlim=[0, 1], ylim=[0, 1.02])
    ax1.legend(); ax1.grid(alpha=0.3)

    # PR decoration
    ax2.set(title="Precision-Recall Curves",
            xlabel="Recall", ylabel="Precision",
            xlim=[0, 1], ylim=[0, 1.02])
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/roc_pr_curves.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"📈  {path} saved")


# ─── 5. Model comparison table ────────────────────────────
def compare_models(all_metrics: list):
    df = pd.DataFrame(all_metrics).set_index("model")
    df = df.sort_values("roc_auc", ascending=False)

    print("\n" + "="*70)
    print("  MODEL COMPARISON TABLE")
    print("="*70)
    print(df.to_string(float_format="{:.4f}".format))
    print("="*70)

    best = df.index[0]
    print(f"\n🏆  Best model: {best}  (ROC-AUC = {df.loc[best,'roc_auc']:.4f})")

    # Heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(df[["accuracy","precision","recall","f1","roc_auc","pr_auc"]],
                annot=True, fmt=".4f", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Model Performance Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.show()

    df.to_csv(f"{PLOTS_DIR}/model_comparison.csv")
    return df


# ─── 6. Grad-CAM ──────────────────────────────────────────
class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: keras.Model, last_conv_layer_name: str):
        self.model     = model
        self.layer     = last_conv_layer_name
        self.grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output,
                     model.output],
        )

    def compute(self, img_array: np.ndarray):
        """img_array: (1, H, W, 3) normalised."""
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(img_array)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap  = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap  = tf.squeeze(heatmap).numpy()
        heatmap  = np.maximum(heatmap, 0)
        heatmap  = heatmap / (heatmap.max() + 1e-8)
        return heatmap

    def overlay(self, img_array, heatmap, alpha=0.4):
        import cv2
        heatmap_resized = cv2.resize(heatmap,
                                     (img_array.shape[1], img_array.shape[0]))
        heatmap_colored = (plt.cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
        overlay = (alpha * heatmap_colored + (1 - alpha) * img_array * 255).astype(np.uint8)
        return overlay


def visualise_gradcam(model, val_gen, last_conv_layer: str,
                      n: int = 8, model_name: str = "model"):
    cam = GradCAM(model, last_conv_layer)
    val_gen.reset()
    batch_imgs, batch_labels = next(val_gen)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    fig.suptitle(f"Grad-CAM Visualisation – {model_name}",
                 fontsize=14, fontweight="bold")

    for i in range(n):
        img   = batch_imgs[i:i+1]
        label = int(batch_labels[i])
        pred  = float(model.predict(img, verbose=0)[0][0])
        hm    = cam.compute(img)
        ov    = cam.overlay(batch_imgs[i], hm)

        axes[0, i].imshow(batch_imgs[i])
        axes[0, i].set_title(f"GT: {CLASSES[label]}", fontsize=9)
        axes[1, i].imshow(hm, cmap="jet")
        axes[1, i].set_title("Heatmap", fontsize=9)
        axes[2, i].imshow(ov)
        pred_label = CLASSES[int(pred >= THRESHOLD)]
        color = "green" if pred_label == CLASSES[label] else "red"
        axes[2, i].set_title(f"Pred:{pred_label}\n{pred:.2f}",
                              fontsize=8, color=color)

        for ax in axes[:, i]:
            ax.axis("off")

    plt.tight_layout()
    path = f"{PLOTS_DIR}/{model_name}_gradcam.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"🔍  {path} saved")


# ─── 7. Full evaluation pipeline ──────────────────────────
def evaluate_all_models(model_paths: dict, val_gen):
    """
    model_paths: {name: path_to_saved_model}
    """
    all_metrics = []
    roc_data    = {}

    for name, path in model_paths.items():
        print(f"\n🔄  Loading {name} from {path} …")
        model = keras.models.load_model(path)

        y_true, y_proba, y_pred = get_predictions(model, val_gen)
        metrics = compute_metrics(y_true, y_proba, y_pred, name)
        all_metrics.append(metrics)
        roc_data[name] = (y_true, y_proba)

        plot_confusion_matrix(y_true, y_pred, name)

    plot_roc_pr(roc_data)
    df = compare_models(all_metrics)
    return df


if __name__ == "__main__":
    print("Run evaluate_all_models(model_paths, val_gen) after training.")
