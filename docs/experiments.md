# Experiments

This document summarizes selected local and remote training experiments.

## YOLOv12n local CPU baseline

Date: 2026-04-25  
Environment: Apple M4 Pro, CPU  
Model: `yolov12n.yaml`  
Dataset: Roboflow Pothole Detection Dataset v2  
Image size: 416  
Batch size: 2  
Epochs: 40  
Device: CPU  

### Validation results

| Metric | Value |
|---|---:|
| Precision | 0.687 |
| Recall | 0.576 |
| mAP50 | 0.642 |
| mAP50-95 | 0.311 |

### Notes

This run was performed locally as a CPU-based baseline. It is suitable for validating the end-to-end workflow and producing a usable inference model for the Gradio demo.

Full training should still be performed on a CUDA-enabled machine for better speed and more extensive experimentation.

## YOLOv12n local CPU baseline — test evaluation

Date: 2026-04-26  
Environment: Apple M4 Pro, CPU  
Model: `yolov12n.yaml`  
Weights: `weights/local/yolov12n_cpu_40e_416_b2_best.pt`  
Dataset: Roboflow Pothole Detection Dataset v2  
Split: test  
Image size: 416  
Batch size: 2  
Device: CPU  

### Test results

| Metric | Value |
|---|---:|
| Precision | 0.750 |
| Recall | 0.611 |
| mAP50 | 0.696 |
| mAP75 | 0.312 |
| mAP50-95 | 0.348 |

### Notes

The model was trained locally on CPU for 40 epochs using `yolov12n.yaml`, image size 416, and batch size 2.

The evaluation was performed with `scripts/evaluate_yolov12.py` on the test split. Model weights and generated evaluation outputs are stored locally and are not committed to the repository.


## YOLOv12n local CPU baseline — extended training

Date: 2026-04-26  
Environment: Apple M4 Pro, CPU  
Model: `yolov12n.yaml`  
Weights: `weights/local/yolov12n_cpu_100e_416_b2_best.pt`  
Dataset: Roboflow Pothole Detection Dataset v2  
Split: test  
Image size: 416  
Batch size: 2  
Device: CPU  

### Test results

| Metric | Previous 40e | Extended 100e |
|---|---:|---:|
| Precision | 0.750 | 0.818 |
| Recall | 0.611 | 0.723 |
| mAP50 | 0.696 | 0.779 |
| mAP50-95 | 0.348 | 0.445 |

### Notes

This model was obtained by continuing local CPU training from the previous 40-epoch baseline. The extended training improved both detection confidence and localization quality, especially on mAP50-95.

The model weights are stored locally and are not committed to the repository.
