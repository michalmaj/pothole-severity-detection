# Pothole Severity Detection

Computer vision project for pothole detection with severity-oriented analysis, training pipelines, and evaluation workflows.

> Project status: early development.

## Overview

This repository is an educational and portfolio-oriented computer vision project focused on detecting potholes in road images and preparing a workflow for severity-oriented analysis.

The initial dataset contains single-class YOLO annotations for pothole detection. Since the dataset does not provide separate severity labels, severity estimation will be treated as a separate analysis stage rather than a direct object detection class at the beginning of the project.

## Dataset

Source: Roboflow Universe  
Dataset: Pothole Detection Dataset v2  
Format: YOLO-compatible annotations

| Split | Images |
|---|---:|
| Train | 1037 |
| Validation | 296 |
| Test | 149 |
| Total | 1482 |

Initial class configuration:

| Class ID | Class Name |
|---:|---|
| 0 | pothole |

## Goals

- Prepare a clean and reproducible computer vision project structure.
- Train an object detection model for pothole detection.
- Evaluate model performance using standard detection metrics.
- Analyze pothole severity using additional post-processing or future model extensions.
- Build a GitHub-ready workflow with clear documentation, experiments, and reproducible setup.

## Planned Workflow

1. Project setup and environment management.
2. Dataset preparation and validation.
3. Exploratory data analysis.
4. Baseline object detection training.
5. Model evaluation.
6. Inference pipeline.
7. Severity-oriented analysis.
8. Optional MLOps improvements.

## Installation

This project uses `uv` for Python environment and dependency management.


## Local smoke training

The local smoke training workflow verifies that the dataset, YOLOv12 model initialization, training loop, validation, and weight export work end to end.

Before running the script, place the dataset under:

```text
data/pothole_detection_v2/
├── data.yaml
├── train/
├── valid/
└── test/
```

Run a minimal CPU-based smoke training:

```bash
uv run python scripts/train_yolov12_smoke.py
```

The default smoke test uses a small configuration:

model: yolov12n.yaml
device: cpu
epochs: 1
image size: 320
batch size: 1

On Apple Silicon, MPS is detected, but YOLOv12 training may fail depending on PyTorch and Ultralytics compatibility. CPU smoke training is supported. Full training is recommended on a CUDA-enabled machine.

## Gradio inference app

The project includes a Gradio application for running pothole detection and severity-oriented visualization on images and short videos.

By default, the app expects model weights at:

```text
runs/local_smoke/yolov12_smoke/weights/best.pt
```

Run the app:

```bash
uv run python scripts/run_gradio_app.py
```

Open the local URL printed in the terminal, usually:

```text
http://127.0.0.1:7860/
```

To use a custom model path:

```bash
MODEL_PATH=weights/local/yolov12n_cpu_40e_416_b2_best.pt \
uv run python scripts/run_gradio_app.py
```

The YOLOv12 model detects potholes. Severity labels are estimated using a post-processing heuristic based on bounding box size and vertical position in the image.