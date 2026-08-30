# Experiment configuration

Every file in `configs/experiments/` is a complete, self-contained description
of one training / evaluation / prediction experiment. The filename is the
human-readable experiment identifier.

The files are parsed and validated by
`pothole_severity_detection.experiment_config.load_experiment_config`, which the
workflow scripts (`scripts/train_yolov12.py`, `scripts/evaluate_yolov12.py`,
`scripts/predict_yolov12.py`, `scripts/inspect_experiment_config.py`) call. To
check a file against the schema:

```bash
uv run python scripts/inspect_experiment_config.py \
  --config configs/experiments/<name>.yaml
```

## Sections

| Section | Required | Purpose |
|---|---|---|
| `experiment` | yes | `name` (the experiment id), optional `description`, `type` |
| `dataset` | yes | `data_yaml` (path to the Ultralytics dataset YAML, must end `.yaml`); optional `name`, `classes` |
| `model` | yes | `source` (an Ultralytics model config name such as `yolov12n.yaml`, or a `.pt` weights path); optional `architecture`, `initial_weights`, `output_weights` |
| `training` | yes | device, epochs, image size, batch size, workers, AMP, augmentation overrides |
| `evaluation` | no | `split` (`train`/`val`/`test`), and optional `image_size` / `batch_size` / `device` that override the `training` values for evaluation |
| `prediction` | no | `source`, `output_dir`, `confidence`, `recursive` |
| `outputs` | no | `training_project`, `training_name`, `evaluation_dir`, `exist_ok` |
| `notes` | no | free-text list, informational |

Unknown sections and unknown keys within a section are rejected.

## `epochs` vs `additional_epochs`

Exactly one must be set under `training`:

- `epochs` — train for this many epochs from `model.source`.
- `additional_epochs` — continue training from existing weights
  (`model.source` pointing at a `.pt` file) for this many more epochs. Used for
  fine-tuning experiments. `total_effective_epochs` may be recorded alongside it
  as documentation; it is not consumed.

`seed` (default `0`) and `deterministic` (default `true`) under `training:` are
passed straight through to the Ultralytics train/val call and recorded in the
result record.

## Augmentation overrides

Optional keys under `training:` — `scale`, `mosaic`, `mixup`, `copy_paste`,
`hsv_h`, `hsv_s`, `hsv_v`, `degrees`, `translate`, `shear`, `perspective`,
`fliplr`, `flipud`, `close_mosaic`, `optimizer`. Any key that is set is passed
straight through to the Ultralytics training call; any key left out uses the
Ultralytics default.

## `evaluation.metrics` (removed)

Older configs carried a hand-written `evaluation.metrics` block. It has been
removed. Evaluation results are now committed as machine-readable records under
`docs/results/`, written by `scripts/evaluate_yolov12.py`. A config that still
contains `evaluation.metrics` is rejected by the schema.

Generated datasets, weights, logs, runs, reports, and prediction outputs are
kept local and are not committed to Git.
