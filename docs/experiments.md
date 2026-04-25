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
