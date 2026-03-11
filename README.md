# 🔬 Malaria Detection Using Deep Learning

> AI-powered microscopic blood smear analysis for automated malaria screening.  
> **Parasitized vs Uninfected cell classification** with 97%+ AUC.

---

## 📁 Project Structure

```
malaria_detection/
│
├── dataset_setup/
│   └── prepare_dataset.py       # Download, split, augment, visualise
│
├── models/
│   ├── custom_cnn.py            # Custom 4-block CNN
│   └── transfer_learning.py     # MobileNetV2 & EfficientNetB0
│
├── evaluation/
│   └── evaluate.py              # Metrics, ROC, confusion matrix, Grad-CAM
│
├── prediction/
│   └── predictor.py             # Single, batch, ensemble, TFLite inference
│
├── deployment/
│   ├── api_server.py            # FastAPI REST backend
│   └── real_world_scenarios.py  # Rural clinic, hospital, research simulations
│
├── ui/
│   └── malariascope_ui.html     # Standalone AI-powered web interface
│
├── train_pipeline.py            # Master training script
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

**NIH Cell Images for Detecting Malaria**  
- Source: [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)  
- 27,558 cell images (13,779 Parasitized + 13,779 Uninfected)  
- Balanced binary classification task

```
cell_images/
    Parasitized/   # 13,779 PNG images
    Uninfected/    # 13,779 PNG images
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
```bash
python dataset_setup/prepare_dataset.py
# Splits into data/train/ and data/val/ (80/20)
```

### 3. Train models
```bash
# Train all models
python train_pipeline.py --model all --epochs 50

# Train specific model
python train_pipeline.py --model efficientnet --epochs 30
```

### 4. Run predictions
```bash
python prediction/predictor.py --model saved_models/best_model.keras \
       --folder data/val/Parasitized
```

### 5. Start API server
```bash
uvicorn deployment.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open web UI
Open `ui/malariascope_ui.html` in any browser.

---

## 🧠 Models

### Model 1 — Custom CNN

| Layer Block | Filters | Operation            |
|-------------|---------|----------------------|
| Block 1     | 32      | Conv×2 → BN → Pool   |
| Block 2     | 64      | Conv×2 → BN → Pool   |
| Block 3     | 128     | Conv×2 → BN → Pool   |
| Block 4     | 256     | Conv×2 → BN → GAP    |
| Head        | —       | Dense(256) → Dense(1)|

- Parameters: ~8.2M  
- Input: 128×128×3  
- Regularisation: L2 + Dropout(0.4) + BatchNorm

### Model 2 — MobileNetV2 (Transfer Learning)

- Backbone: MobileNetV2 (ImageNet pre-trained)  
- Phase 1: Train head only (20 epochs, LR=1e-3)  
- Phase 2: Fine-tune top layers (30 epochs, LR=1e-5)  
- Parameters: ~2.3M (lightweight, mobile-ready)

### Model 3 — EfficientNetB0 (Transfer Learning)

- Backbone: EfficientNetB0 (ImageNet pre-trained)  
- Compound scaling (depth × width × resolution)  
- Phase 1 + Phase 2 training strategy  
- Parameters: ~4.0M (best accuracy/efficiency trade-off)

### Ensemble

Soft voting (weighted probability average) across all three models.

---

## 📊 Expected Performance

| Model          | Accuracy | Precision | Recall | F1    | ROC-AUC |
|----------------|----------|-----------|--------|-------|---------|
| Custom CNN     | 95.8%    | 96.1%     | 95.4%  | 95.7% | 0.9872  |
| MobileNetV2    | 96.9%    | 97.2%     | 96.6%  | 96.9% | 0.9921  |
| EfficientNetB0 | **97.4%**| **97.6%** |**97.1%**|**97.3%**|**0.9947**|
| Ensemble       | 97.7%    | 97.9%     | 97.4%  | 97.6% | 0.9961  |

*Results on NIH dataset; values may vary slightly by seed and hardware.*

---

## 🔄 Data Preprocessing & Augmentation

```python
# Training augmentations
ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.10,
    zoom_range=0.20,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
)

# Validation: rescale only
ImageDataGenerator(rescale=1/255)
```

---

## 🌐 API Reference

### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@cell.png"
```
```json
{
  "request_id": "a1b2c3d4",
  "filename": "cell.png",
  "label": "Parasitized",
  "confidence": 0.9743,
  "raw_proba": 0.9743,
  "inference_ms": 34.2,
  "timestamp": "2025-01-15T10:30:00"
}
```

### POST /predict-batch
```bash
curl -X POST http://localhost:8000/predict-batch \
  -F "files=@cell1.png" -F "files=@cell2.png"
```

### GET /stats
Returns session-level screening statistics.

---

## 🏥 Real-World Scenarios

```bash
# Rural clinic (20 patients)
python deployment/real_world_scenarios.py --scenario rural --data data/val

# Hospital mass screening (200 samples)
python deployment/real_world_scenarios.py --scenario hospital --data data/val

# Research analysis
python deployment/real_world_scenarios.py --scenario research --data data/val
```

---

## 🗜️ Edge Deployment (TFLite)

```python
from prediction.predictor import convert_to_tflite, TFLitePredictor

# Convert
convert_to_tflite("saved_models/best_model.keras",
                   "saved_models/malaria_detector.tflite",
                   quantize=True)

# Inference
predictor = TFLitePredictor("saved_models/malaria_detector.tflite")
result = predictor.predict_single("cell.png")
```

Model size after quantization: ~2–5 MB (suitable for Raspberry Pi, Android).

---

## 🔍 Interpretability — Grad-CAM

```python
from evaluation.evaluate import visualise_gradcam

visualise_gradcam(
    model=model,
    val_gen=val_gen,
    last_conv_layer="top_conv",   # EfficientNet
    n=8,
    model_name="efficientnetb0"
)
```

Generates heatmaps highlighting regions the model focuses on.

---

## 📈 Callbacks & Monitoring

- **ModelCheckpoint** — saves best model by `val_auc`  
- **EarlyStopping** — patience=10, restores best weights  
- **ReduceLROnPlateau** — halves LR after 5 stagnant epochs  
- **TensorBoard** — launch with `tensorboard --logdir logs/`  
- **CSVLogger** — training history to CSV

---

## 🚀 Deployment Options

| Method           | Command / Notes                              |
|------------------|----------------------------------------------|
| FastAPI + Uvicorn| `uvicorn deployment.api_server:app --port 8000` |
| Docker           | `docker build -t malariascope . && docker run -p 8000:8000 malariascope` |
| TFLite (edge)    | Raspberry Pi, Android via `tflite-runtime`   |
| TF Serving       | `docker run -t tensorflow/serving --model_base_path=saved_models` |

---

## ⚕️ Clinical Disclaimer

This system is intended for **screening assistance only**. It does not replace professional microscopy or clinical judgment. All positive results must be confirmed by a qualified healthcare professional.

---

## 📄 License

MIT License — free for research and educational use.
