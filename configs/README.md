# Configuration Files

This directory contains project configuration files.

At this stage, experiment configuration files are used as human-readable records
of training and evaluation settings. The training scripts will be extended later
to read these YAML files directly.

## Structure

```text
configs/
└── experiments/
    ├── yolov12n_cpu_smoke.yaml
    ├── yolov12n_cpu_40e_416_b2.yaml
    └── yolov12n_cpu_100e_416_b2.yaml
```
Generated outputs, model weights, datasets, and logs are not committed to Git.
