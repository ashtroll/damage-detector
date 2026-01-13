<div align="center">

<h1>🚗 Car Damage Detection & Visualization Suite</h1>
<p><strong>Deep-learning powered toolkit to detect, localize and analyze automotive exterior damage (scratches, dents, rust, paint defects) using YOLOv8 and Faster R‑CNN, plus a desktop GUI for interactive review.</strong></p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Features](#1-high-level-overview) • [Installation](#6-installation--quick-start) • [Training](#4-model-training--evaluation) • [Usage](#5-application-desktop-gui-deep-dive) • [Contributing](CONTRIBUTING.md)

</div>

---

## 1. High-Level Overview

This repository combines three pillars:

| Pillar | Purpose | Technologies |
|--------|---------|--------------|
| Data & Annotation | Curated, polygon‑annotated dataset of car body damage | Roboflow, manual QC |
| Model Training | Comparative experiments (YOLOv8 vs Faster R‑CNN) | Ultralytics YOLO, Detectron2 concepts (Faster R-CNN) |
| Interactive App | Lightweight desktop image review & inference | Tkinter, OpenCV, Pillow, ultralytics |

The goal: provide a reproducible workflow from raw images → cleaned + annotated dataset → trained model → end‑user visualization.

---

## 2. Repository Structure & Components

```
Car-damage-detection/
├── damage_detection_app/
│   ├── app.py               # Tkinter GUI: folder ingestion, inference, display, logging.
│   ├── requirements.txt     # Minimal runtime dependencies for the GUI.
│   ├── README.md            # App-specific quick start.
│   └── test/                # Sample images for validation/demo.
├── img/                     # Metrics plots, screenshots, comparison visuals, dataset examples.
├── training/
│   ├── YOLOv8.ipynb         # End-to-end YOLOv8 training & evaluation pipeline.
│   └── Faster R-CNN.ipynb   # Faster R-CNN experimentation (Detectron-style workflow).
├── notebooks/official/custom/sdk-custom-image-classification-online.ipynb # (Example Azure/SDK style notebook placeholder)
└── README.md                # (This file) Full project documentation.
```

### 2.1 `damage_detection_app/app.py` – GUI Architecture
The GUI centers around a single `YOLOApp` class:

| Area | Description |
|------|-------------|
| Initialization | Creates frames (top for canvas, bottom for controls + log), configures drag‑and‑drop (TkinterDnD), sets window styling and graceful close handler. |
| Model Loading | Lazily loads `model/best.pt` on first folder open or drag event (`load_model`). Replace this file with your own trained weights. |
| Image Ingestion | Folder selection (dialog or drag) filters by extension (`png`, `jpg`, `jpeg`) and stores paths in `image_list`. Navigation handled by `show_prev_image` / `show_next_image`. |
| Inference Pipeline | `detect_objects` calls the ultralytics `YOLO` model: results tensor → counts labels → plots annotated image (bounding boxes + labels) using built‑in `.plot()`. |
| Rendering | OpenCV BGR→RGB conversion → Pillow resize maintaining aspect ratio → Tkinter `Canvas` redraw on resize (`on_resize`). |
| Logging | `update_log` aggregates detections (group counts of each class) and writes formatted bullet list to a right‑hand `Text` widget. |
| UX Enhancements | Placeholder clickable text for “Choose a directory”, drag‑and‑drop of folders, responsive scaling, confirmation dialog on exit. |

### 2.2 Notebooks
| Notebook | Focus | Key Outputs |
|----------|-------|-------------|
| `training/YOLOv8.ipynb` | Download dataset, preprocess, configure hyperparameters (batch size sweep), train, validate, export weights (`best.pt`) | Precision curve, loss curves, mAP50, confusion matrix, final weights |
| `training/Faster R-CNN.ipynb` | Alternative architecture test; plays with backbone depth (R50 vs R101), LR scheduling, solver steps to combat overfitting | Loss progression, mAP50 comparison, confusion matrix, qualitative eval |
| `notebooks/official/...` | (Placeholder / external integration example) | Demonstrates potential cloud / SDK workflow |

### 2.3 Images & Metrics (`img/`)
Stored plots document evaluation, aiding regression tracking when retraining.

### 2.4 Test Images (`damage_detection_app/test/`)
Small curated subset to sanity‑check inference quality & GUI rendering.

---

## 3. Dataset & Annotation Strategy

Classes (example): scratches, rust, paint fading, paint cracks, dents, structural cracks, PDR dents.

| Version | Image Count | Augmentation | Annotation Geometry |
|---------|-------------|--------------|---------------------|
| Base | 456 | None | Polygons (fine-grained boundaries) |
| Augmented | 1140 | Flip, rotate, saturation jitter, cutout | Inherited polygons |

Annotation rationale: polygon labeling reduces background bleed into bounding boxes (important for subtle paint defects). Time investment improves precision especially with limited dataset size.

Dataset source & details: [Roboflow Project Link](https://universe.roboflow.com/cardetecion/car-paint-damage-detection)

### 3.1 Augmentation Techniques
- Horizontal & vertical flips
- Rotation: ±15°
- Saturation adjustment: −35% to +35%
- Cutout: 10 masks (~2% each) to encourage robustness to occlusion



### 3.2 Class Distribution & Visual Reference


---

## 4. Model Training & Evaluation

**Our Training Configuration:**
- **Model**: YOLOv8s (small variant for 4GB VRAM)
- **Batch Size**: 16
- **Epochs**: 100 (early stopping at best mAP)
- **Image Size**: 640x640
- **Workers**: 2
- **Device**: NVIDIA RTX 3050 (4GB)
- **Training Time**: ~6-7 hours total

### 4.1 YOLOv8 Experiments
Batch sizes explored: `-1 (auto)`, 8, 16, 32 across augmented vs non‑augmented data using `yolov8m` pretrained weights.

Key insight: auto batch (`-1`) + augmentation delivered best stability and generalization.

<details>
<summary>Metrics (best run)</summary>

| Metric | Observations |
|--------|--------------|
| Precision | Steady climb, early convergence |
| Total Loss | Multi-component decline (cls, box, dfl) visible in combined curve |
| mAP50 | Competitive given limited data size |
| Confusion Matrix | Low cross‑class leakage on high contrast damage types |

</details>

## 5. Application (Desktop GUI) Deep Dive

### 5.1 Runtime Flow
1. User selects or drags a folder → file list built.
2. Lazy model load from `model/best.pt` (ensure this file exists; place exported YOLO weights there).
3. For each image navigation event: read with OpenCV → inference → annotate via `.plot()` → convert to Pillow → resize → render on Tkinter Canvas.
4. Detection tensor processed to aggregate label counts → written to log pane.

### 5.2 Dependencies Rationale
| Library | Role |
|---------|-----|
| `opencv-python` | Fast image I/O and color conversion |
| `pillow` | High-quality resizing & Tkinter compatibility |
| `tkinterdnd2` | Drag & drop UX for directory ingest |
| `ultralytics` | Model loading, inference, annotation rendering |

### 5.3 Extending the App
| Goal | What to Change |
|------|----------------|
| New model version | Replace `model/best.pt` with new weights; keep same path or update `load_model`. |
| Add confidence threshold | Insert filtering in `detect_objects` before `update_log`. |
| Export results | Save `results[0].boxes.data` to CSV/JSON after inference. |
| Support video | Iterate frames from `cv2.VideoCapture` → reuse detection/display pipeline. |
| Multi-class color coding | Modify `.plot()` output or overlay custom rectangles using class→color map. |

### 5.4 Error Handling & UX
Errors (missing model, unreadable image) are appended to the log panel and stack traces printed (for dev). Production hardening could suppress raw tracebacks and show modal dialogs.

### 5.5 Known Limitations
- Model path hardcoded; no config file yet.
- No batch evaluation metrics inside GUI.
- Scaling prioritizes fit; no zoom/pan.
- Confidence & NMS parameters not user-adjustable in UI.

---

## 6. Installation & Quick Start

### 6.1 Clone Repository
```bash
git clone https://github.com/ashtroll/car-damage-detection-deep-learning.git
cd car-damage-detection-deep-learning/damage_detection_app
```

### 6.2 (Optional) Create Virtual Environment – Windows (cmd)
```cmd
python -m venv .venv
".venv\Scripts\activate"
```

### 6.3 Install Dependencies
```cmd
pip install -r requirements.txt
```

### 6.4 Place Model Weights
Download the trained model from [Releases](https://github.com/ashtroll/car-damage-detection-deep-learning/releases) and place it in the model directory:
```cmd
mkdir model
# Download best.pt from GitHub Releases, then:
copy path\to\downloaded\best.pt model\best.pt
```

### 6.5 Run GUI
```cmd
python app.py
```

Drag a folder of images into the window or click “Open Folder”. Use arrow buttons to navigate.

### 6.6 Training (Optional)
Open notebooks under `training/` in Colab: upload Roboflow export, adjust paths, run cells, download `best.pt`.

---

## 7. Evaluation & Reproducibility Notes
| Aspect | Recommendation |
|--------|----------------|
| Hardware | Use GPU (Colab T4/A100) for YOLO; Faster R-CNN needs more VRAM for larger backbones. |
| Random Seeds | Fix seeds (`torch`, `numpy`) for more stable comparisons when altering batch sizes. |
| Version Pinning | Track ultralytics version (see `requirements.txt`); major updates can alter label assignment. |
| Metrics | Prefer mAP50-95 for richer view; current focus on mAP50 due to dataset scale. |

---

## 8. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: ultralytics` | Env not activated / deps missing | Activate venv + reinstall requirements |
| Empty detections | Wrong weights / incompatible classes | Verify class names in training and model export |
| Blurry resized image | Low original resolution | Enable high-quality interpolation already set (LANCZOS) – no action |
| GUI freeze on huge folders | Large I/O on main thread | Add threading or prefetch caching |
| Model not found | Missing `model/best.pt` | Place weights file; check relative path |

---

## 9. Future Improvements
1. Config file (`config.yaml`) for model path, thresholds.
2. Batch inference & CSV/JSON export.
3. Confidence/NMS sliders in GUI.
4. Lightweight web version (FastAPI + simple React front-end).
5. Active learning loop: feed low-confidence predictions back into annotation queue.
6. ONNX / TensorRT export for speed on edge devices.

---

## 10. License & Use
Dataset annotations produced manually; ensure any redistribution complies with original image source rights. Code is provided for educational and research use—add an explicit license file (MIT/Apache) if redistribution terms need clarification.

---

## 11. Attributions
- Ultralytics YOLO for core detection engine.
- Roboflow for dataset hosting & augmentation pipeline.
- TkinterDnD2 for drag‑and‑drop integration.
- OpenCV & Pillow for image handling.

---

## 12. Quick Reference (Cheat Sheet)
| Action | Command (Windows cmd) |
|--------|-----------------------|
| Create venv | `python -m venv .venv` |
| Activate venv | `".venv\Scripts\activate"` |
| Install deps | `pip install -r requirements.txt` |
| Run app | `python app.py` |
| Add weights | `copy path\to\best.pt model\best.pt` |

---

## 13. Visual Demo
<img src="img/application_demo.gif" alt="Application Demo" width="700" />

----

## 14. Summary
This project demonstrates a complete, minimal yet extensible pipeline for car exterior damage detection—from curated polygon annotations through model benchmarking to a user‑friendly desktop visualization tool. YOLOv8 currently offers the best trade‑off on the constrained dataset; the architecture choices, augmentation strategy, and GUI design aim to be transparent so you can iterate further.

Feel free to open issues or submit PRs for enhancements..
