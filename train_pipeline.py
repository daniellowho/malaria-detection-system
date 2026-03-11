"""
=============================================================
 Malaria Detection – Master Training Pipeline
=============================================================
Run:  python train_pipeline.py [--model all|cnn|mobilenet|efficientnet]
                                [--data  data/]
                                [--epochs 50]
=============================================================
"""

import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))
from dataset_setup.prepare_dataset   import build_generators, build_tf_datasets, IMG_SIZE
from models.custom_cnn               import build_custom_cnn, compile_model, get_callbacks, plot_history, BATCH_SIZE
from models.transfer_learning        import MobileNetV2Detector, EfficientNetB0Detector
from evaluation.evaluate             import evaluate_all_models, compare_models


# ─── GPU setup ────────────────────────────────────────────
def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅  GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("⚠️   No GPU found — using CPU (training will be slow)")


# ─── Reproducibility ──────────────────────────────────────
def set_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Train Custom CNN ─────────────────────────────────────
def run_custom_cnn(train_gen, val_gen, epochs):
    from models.custom_cnn import build_custom_cnn, compile_model, get_callbacks, plot_history, MODEL_NAME
    print("\n" + "━"*55)
    print("  Training  ▸  Custom CNN")
    print("━"*55)

    model   = compile_model(build_custom_cnn())
    t0      = time.time()
    history = model.fit(
        train_gen, epochs=epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(MODEL_NAME),
        verbose=1,
    )
    elapsed = time.time() - t0
    model.save(f"saved_models/{MODEL_NAME}.keras")
    plot_history(history, MODEL_NAME)
    print(f"⏱  Custom CNN training time: {elapsed/60:.1f} min")
    return model, history


# ─── Train MobileNetV2 ────────────────────────────────────
def run_mobilenetv2(train_gen, val_gen, epochs):
    print("\n" + "━"*55)
    print("  Training  ▸  MobileNetV2")
    print("━"*55)
    det = MobileNetV2Detector()
    det.build()
    t0  = time.time()
    det.train_phase1(train_gen, val_gen, epochs=min(20, epochs//2))
    det.train_phase2(train_gen, val_gen, epochs=epochs - min(20, epochs//2))
    print(f"⏱  MobileNetV2 training time: {(time.time()-t0)/60:.1f} min")
    return det.model


# ─── Train EfficientNetB0 ─────────────────────────────────
def run_efficientnetb0(train_gen, val_gen, epochs):
    print("\n" + "━"*55)
    print("  Training  ▸  EfficientNetB0")
    print("━"*55)
    det = EfficientNetB0Detector()
    det.build()
    t0  = time.time()
    det.train_phase1(train_gen, val_gen, epochs=min(20, epochs//2))
    det.train_phase2(train_gen, val_gen, epochs=epochs - min(20, epochs//2))
    print(f"⏱  EfficientNetB0 training time: {(time.time()-t0)/60:.1f} min")
    return det.model


# ─── Main pipeline ────────────────────────────────────────
def main(args):
    configure_gpu()
    set_seeds()
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs",         exist_ok=True)
    os.makedirs("plots",        exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Malaria Detection – Training Pipeline")
    print(f"{'='*55}")
    print(f"  Model  : {args.model}")
    print(f"  Data   : {args.data}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {BATCH_SIZE}")
    print(f"  ImgSz  : {IMG_SIZE}")

    # Build data generators
    print("\n📂  Building data generators …")
    train_gen, val_gen = build_generators(args.data)

    trained = {}

    if args.model in ("all", "cnn"):
        model, hist = run_custom_cnn(train_gen, val_gen, args.epochs)
        trained["custom_cnn"] = "saved_models/custom_cnn_best.keras"

    if args.model in ("all", "mobilenet"):
        run_mobilenetv2(train_gen, val_gen, args.epochs)
        trained["mobilenetv2"] = "saved_models/mobilenetv2_best.keras"

    if args.model in ("all", "efficientnet"):
        run_efficientnetb0(train_gen, val_gen, args.epochs)
        trained["efficientnetb0"] = "saved_models/efficientnetb0_best.keras"

    # Evaluate all trained models
    if trained:
        print("\n\n📊  Running evaluation …")
        df = evaluate_all_models(trained, val_gen)

        best_name = df.index[0]
        best_path = trained[best_name]
        import shutil
        shutil.copy(best_path, "saved_models/best_model.keras")
        print(f"\n🏆  Best model: {best_name}")
        print(f"    Copied → saved_models/best_model.keras")

    print("\n✅  Pipeline complete!\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Malaria Detection Training Pipeline")
    ap.add_argument("--model",  default="all",
                    choices=["all", "cnn", "mobilenet", "efficientnet"],
                    help="Which model(s) to train")
    ap.add_argument("--data",   default="data",
                    help="Root of organised dataset (train/ val/ subdirs)")
    ap.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs")
    main(ap.parse_args())
