"""Gradio application for pothole detection and severity heuristic visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from pothole_severity_detection.inference.detector import (
    detect_media,
    is_image_file,
    is_video_file,
)
from pothole_severity_detection.inference.model_loader import (
    load_model,
    resolve_model_path,
)

model: Any | None = None


def get_model() -> Any:
    """Load the model lazily on the first inference request."""
    global model

    if model is None:
        model = load_model()

    return model


def get_model_status_markdown() -> str:
    """Create a small model status section for the Gradio interface."""
    model_path = resolve_model_path()
    availability = "Available" if model_path.exists() else "Missing"

    return f"""
### Loaded model configuration

| Field | Value |
|---|---|
| Default model path | `{model_path}` |
| Status | **{availability}** |
| Override mechanism | `MODEL_PATH=/path/to/best.pt` |

The application loads the model lazily on the first inference request. If the
default weights are missing, run training first or provide a custom `MODEL_PATH`.
"""


def process_media(input_path: str | None, confidence: float):
    """Process uploaded image or video and return Gradio outputs."""
    if input_path is None:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            None,
            "Upload an image or video file.",
        )

    path = Path(input_path)

    try:
        output_path = detect_media(
            model=get_model(),
            input_path=path,
            confidence=confidence,
        )
    except Exception as error:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            None,
            f"Error: {error}",
        )

    if is_video_file(output_path):
        return (
            gr.update(value=None, visible=False),
            gr.update(value=str(output_path), visible=True),
            str(output_path),
            "Video processed successfully.",
        )

    if is_image_file(output_path):
        return (
            gr.update(value=str(output_path), visible=True),
            gr.update(value=None, visible=False),
            str(output_path),
            "Image processed successfully.",
        )

    return (
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        None,
        "Unsupported output format.",
    )


def build_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="Pothole Detection with Severity Heuristic") as app:
        gr.Markdown(
            """
            # Pothole Detection with Severity Heuristic

            Applied computer vision demo for pothole detection in road images
            and short videos. YOLOv12 predicts pothole bounding boxes, while
            severity is estimated as a transparent post-processing heuristic.
            """
        )

        with gr.Tabs():
            with gr.Tab("Inference"):
                gr.Markdown(
                    """
                    Upload an image or a short video and run detection.

                    The confidence threshold controls how strict the detector is.
                    Lower values may reveal more potholes but can increase false
                    positives. Higher values show fewer, more confident detections.
                    """
                )

                with gr.Row():
                    with gr.Column():
                        input_file = gr.File(
                            label="Upload image or video",
                            type="filepath",
                            file_types=["image", "video"],
                        )
                        confidence = gr.Slider(
                            minimum=0.01,
                            maximum=0.9,
                            value=0.25,
                            step=0.01,
                            label="Confidence threshold",
                        )
                        run_button = gr.Button(
                            "Run detection",
                            variant="primary",
                        )

                    with gr.Column():
                        output_image = gr.Image(
                            label="Detected image",
                            interactive=False,
                            visible=False,
                        )
                        output_video = gr.Video(
                            label="Detected video",
                            interactive=False,
                            visible=False,
                        )
                        output_file = gr.File(label="Download output")
                        status = gr.Textbox(label="Status", interactive=False)

                run_button.click(
                    fn=process_media,
                    inputs=[input_file, confidence],
                    outputs=[output_image, output_video, output_file, status],
                )

            with gr.Tab("Model & Results"):
                gr.Markdown(
                    """
                    ## Model summary

                    The current best documented local baseline is based on
                    `YOLOv12n` and was fine-tuned on CPU from the 100-epoch
                    baseline using gentler augmentation settings.

                    | Model | Split | Precision | Recall | mAP50 | mAP50-95 |
                    |---|---|---:|---:|---:|---:|
                    | YOLOv12n gentle aug | test | 0.805 | 0.778 | 0.836 | 0.490 |

                    The full experiment history is documented in
                    `docs/experiments.md`.
                    """
                )
                gr.Markdown(get_model_status_markdown())

            with gr.Tab("Severity Heuristic"):
                gr.Markdown(
                    """
                    ## Severity heuristic

                    The dataset contains one detection class: `pothole`.
                    It does not contain ground-truth severity labels.

                    Therefore, severity is not learned by the model. It is
                    estimated after detection from:

                    - normalized bounding box area,
                    - vertical position of the bounding box in the image.

                    The heuristic is intentionally simple and interpretable. It
                    should be read as a visual prioritization layer, not as a
                    validated road-damage severity model.
                    """
                )

            with gr.Tab("Limitations"):
                gr.Markdown(
                    """
                    ## Scope and limitations

                    This demo is a detection-first prototype.

                    Current limitations:

                    - the dataset has pothole bounding boxes only,
                    - there are no expert severity labels,
                    - severity is heuristic, not learned,
                    - local training was performed on CPU,
                    - camera calibration, pothole depth, and real-world road
                      context are not modeled.

                    A production-grade severity model would require additional
                    labels and measurements, such as pothole depth, surface area,
                    road context, calibrated camera geometry, or expert severity
                    grades.
                    """
                )

    return app


if __name__ == "__main__":
    build_app().launch()
