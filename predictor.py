"""
=============================================================
 Malaria Detection – Prediction Module
=============================================================
Supports:
  • Single image prediction
  • Batch prediction (folder / list)
  • Ensemble prediction (majority vote + probability avg)
  • TFLite inference (edge deployment)
=============================================================
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict


# ─── Config ───────────────────────────────────────────────
IMG_SIZE   = (128, 128)
THRESHOLD  = 0.50
CLASSES    = {0: "Uninfected", 1: "Parasitized"}


@dataclass
class PredictionResult:
    filename:      str
    label:         str
    confidence:    float          # probability of predicted class
    raw_proba:     float          # P(Parasitized)
    inference_ms:  float
    threshold:     float = THRESHOLD

    @property
    def is_parasitized(self) -> bool:
        return self.label == "Parasitized"

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        bar   = "█" * int(self.confidence * 20)
        empty = "░" * (20 - int(self.confidence * 20))
        color = "🔴" if self.is_parasitized else "🟢"
        return (
            f"{color}  {self.label:<14}  |{bar}{empty}|  "
            f"{self.confidence*100:5.1f}%   ({self.inference_ms:.1f} ms)"
        )


# ─── Preprocessing ────────────────────────────────────────
def preprocess_image(source, target_size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Accepts: file path (str/Path), PIL.Image, or numpy array.
    Returns: (1, H, W, 3) float32 normalised array.
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source).convert("RGB")
    elif isinstance(source, Image.Image):
        img = source.convert("RGB")
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source.astype(np.uint8)).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(source)}")

    img   = img.resize(target_size, Image.LANCZOS)
    arr   = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # (1, H, W, 3)


# ─── Single model predictor ───────────────────────────────
class MalariaPredictor:
    def __init__(self, model_path: str, threshold: float = THRESHOLD,
                 model_name: str = "model"):
        print(f"⏳  Loading model from {model_path} …")
        self.model      = keras.models.load_model(model_path)
        self.threshold  = threshold
        self.model_name = model_name
        self._warmup()
        print(f"✅  {model_name} ready")

    def _warmup(self):
        dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)

    def predict_single(self, source, filename: str = "image") -> PredictionResult:
        arr   = preprocess_image(source)
        t0    = time.perf_counter()
        proba = float(self.model.predict(arr, verbose=0)[0][0])
        ms    = (time.perf_counter() - t0) * 1000

        label      = CLASSES[int(proba >= self.threshold)]
        confidence = proba if label == "Parasitized" else 1.0 - proba

        return PredictionResult(
            filename=str(filename),
            label=label,
            confidence=confidence,
            raw_proba=proba,
            inference_ms=ms,
        )

    def predict_batch(self, sources: list,
                      filenames: Optional[List[str]] = None) -> List[PredictionResult]:
        if filenames is None:
            filenames = [str(i) for i in range(len(sources))]

        results = []
        for src, fn in zip(sources, filenames):
            results.append(self.predict_single(src, fn))
        return results

    def predict_folder(self, folder: str) -> List[PredictionResult]:
        folder   = Path(folder)
        ext      = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
        images   = [f for f in folder.iterdir() if f.suffix.lower() in ext]
        print(f"🔍  Found {len(images)} images in {folder}")
        return self.predict_batch(images, [f.name for f in images])

    def predict_with_report(self, sources, filenames=None, save_json=None):
        if isinstance(sources, (str, Path)) and Path(sources).is_dir():
            results = self.predict_folder(sources)
        elif isinstance(sources, list):
            results = self.predict_batch(sources, filenames)
        else:
            results = [self.predict_single(sources,
                        filenames or "image")]

        # Print report
        print(f"\n{'='*60}")
        print(f"  Prediction Report — {self.model_name}")
        print(f"{'='*60}")
        n_para  = sum(r.is_parasitized for r in results)
        n_unin  = len(results) - n_para
        avg_ms  = np.mean([r.inference_ms for r in results])

        for r in results:
            print(f"  {r.filename:<30}  {r}")

        print(f"\n  Total    : {len(results)}")
        print(f"  🔴 Parasitized  : {n_para}  ({n_para/len(results)*100:.1f}%)")
        print(f"  🟢 Uninfected   : {n_unin}  ({n_unin/len(results)*100:.1f}%)")
        print(f"  Avg inference   : {avg_ms:.1f} ms")
        print(f"{'='*60}\n")

        if save_json:
            with open(save_json, "w") as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
            print(f"📄  Results saved → {save_json}")

        return results


# ─── Ensemble predictor ───────────────────────────────────
class EnsemblePredictor:
    """
    Combines Custom CNN + MobileNetV2 + EfficientNetB0 via
    soft voting (averaged probabilities).
    """

    def __init__(self, model_configs: Dict[str, dict], threshold=THRESHOLD):
        """
        model_configs: {name: {"path": ..., "weight": ...}}
        """
        self.predictors = {}
        self.weights    = {}
        self.threshold  = threshold

        for name, cfg in model_configs.items():
            self.predictors[name] = MalariaPredictor(cfg["path"], name=name)
            self.weights[name]    = cfg.get("weight", 1.0)

        # Normalise weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        print(f"\n🎯  Ensemble ready: {list(self.predictors.keys())}")
        print(f"    Weights: {self.weights}")

    def predict_single(self, source, filename="image") -> PredictionResult:
        t0    = time.perf_counter()
        proba = 0.0

        for name, pred in self.predictors.items():
            r      = pred.predict_single(source, filename)
            proba += r.raw_proba * self.weights[name]

        ms         = (time.perf_counter() - t0) * 1000
        label      = CLASSES[int(proba >= self.threshold)]
        confidence = proba if label == "Parasitized" else 1.0 - proba

        return PredictionResult(
            filename=str(filename),
            label=label,
            confidence=confidence,
            raw_proba=proba,
            inference_ms=ms,
        )

    def predict_batch(self, sources, filenames=None):
        if filenames is None:
            filenames = [str(i) for i in range(len(sources))]
        return [self.predict_single(s, f)
                for s, f in zip(sources, filenames)]


# ─── TFLite inference (edge deployment) ───────────────────
class TFLitePredictor:
    def __init__(self, tflite_path: str, threshold=THRESHOLD):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.inp  = self.interpreter.get_input_details()[0]
        self.out  = self.interpreter.get_output_details()[0]
        self.threshold = threshold

    def predict_single(self, source, filename="image") -> PredictionResult:
        arr = preprocess_image(source).astype(np.float32)
        t0  = time.perf_counter()
        self.interpreter.set_tensor(self.inp["index"], arr)
        self.interpreter.invoke()
        proba = float(self.interpreter.get_tensor(self.out["index"])[0][0])
        ms    = (time.perf_counter() - t0) * 1000

        label      = CLASSES[int(proba >= self.threshold)]
        confidence = proba if label == "Parasitized" else 1.0 - proba
        return PredictionResult(filename=str(filename),
                                label=label, confidence=confidence,
                                raw_proba=proba, inference_ms=ms)


# ─── Model converter ──────────────────────────────────────
def convert_to_tflite(keras_path: str, output_path: str,
                      quantize: bool = True):
    """Convert Keras model to TFLite (optionally int8 quantised)."""
    model     = keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅  TFLite model saved → {output_path}  ({size_kb:.1f} KB)")
    return output_path


# ─── Quick demo ───────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Malaria cell predictor")
    ap.add_argument("--model",  default="saved_models/best_model.keras")
    ap.add_argument("--image",  default=None)
    ap.add_argument("--folder", default=None)
    ap.add_argument("--output", default="predictions.json")
    args = ap.parse_args()

    predictor = MalariaPredictor(args.model)

    if args.image:
        r = predictor.predict_single(args.image, args.image)
        print(r)
    elif args.folder:
        predictor.predict_with_report(args.folder, save_json=args.output)
    else:
        print("Provide --image <path> or --folder <path>")
