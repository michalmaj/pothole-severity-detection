# Severity heuristic

## What this is

Severity is a post-processing heuristic, not a learned output. The dataset
(Roboflow Pothole Detection Dataset v2) contains pothole bounding boxes only --
no ground-truth severity labels -- so the model does not predict severity. After
detection, each box is assigned a `Low` / `Medium` / `High` label from its
geometry. The label is a visual-prioritisation aid, not a severity measurement.

## The formula

For a detected box with pixel area `A_bbox` and vertical centre `y_center`, in a
frame of width `W` and height `H`:

```
score = alpha * (A_bbox / (W * H))  +  (1 - alpha) * (y_center / H)
```

with `alpha = 0.6`.

- The first term is the fraction of the frame the box covers -- apparent size.
- The second term is how far down the frame the box centre sits -- apparent
  proximity, for a roughly forward-facing camera.

Implementation: `severity_score` in
`src/pothole_severity_detection/inference/severity.py`.

## Banding

| score | label |
|---|---|
| `< 0.2` | Low |
| `< 0.4` | Medium |
| otherwise | High |

Implementation: `estimate_severity` in the same module.

## Parameter choices

`alpha` and both thresholds are **chosen defaults, not values calibrated against
data** -- there is no labelled severity data to calibrate against. `alpha = 0.6`
slightly favours apparent size over apparent proximity. The thresholds `0.2` and
`0.4` were set so that all three bands are populated on typical road images.
There is no claim that any of these values is optimal.

In code the three values are `severity.SeverityParameters`
(`alpha`, `low_threshold`, `medium_threshold`); the module-level
`DEFAULT_SEVERITY_PARAMETERS` holds the defaults. Changing them is a research
decision.

## Assumptions

- A roughly forward-facing camera at a consistent height, so vertical position
  in the frame stands in for distance.
- Box area as a size proxy is resolution- and aspect-ratio-dependent.
- A pothole's on-image size conflates its true physical size and its distance
  from the camera.

## Limitations

- Camera-perspective dependence: the vertical-position term assumes a consistent
  viewpoint that real footage will not always match.
- No depth, no 3D geometry, no calibration.
- No ground-truth severity, so the heuristic cannot be validated as-is.
- One global banding is applied regardless of the image.
- The severity layer is independent of the detection model -- changing the
  detector does not change how a given box is scored.

## Possible future validation

These are options for the project owner, not commitments:

- Hand-annotate a small subset (roughly 40-60 potholes) with an expert
  `Low` / `Medium` / `High` grade and measure agreement with the heuristic
  (e.g. Cohen's kappa, a confusion matrix).
- Inspect the score distribution on the test set and check whether the `0.2`
  and `0.4` thresholds land at defensible percentiles.
- Run a sensitivity analysis: how do band assignments shift as `alpha` and the
  thresholds vary?
- A learned severity model would require additional data -- pothole depth,
  surface area, calibrated camera geometry, or expert severity grades.

## References

- `src/pothole_severity_detection/inference/severity.py`
- The "Severity heuristic" section of the project `README.md`
- The "Severity Heuristic" tab in the Gradio demo
