"""Gradio application for pothole detection and severity estimation."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from pothole_severity_detection.inference.detector import (
    is_image_file,
    is_video_file,
    detect_media,
)
from pothole_severity_detection.inference.model_loader import load_model


model = None


def get_model():
    """Load the model lazily on the first request."""
    global model

    if model is None:
        model = load_model()

    return model


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
    with gr.Blocks(title="Pothole Severity Detection") as app:
        gr.Markdown(
            """
            # Pothole Detection & Severity Estimation

            Upload an image or a short video. The YOLOv12 detector finds potholes,
            while severity is estimated using a bounding-box-based heuristic.
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
                    minimum=0.1,
                    maximum=0.9,
                    value=0.4,
                    step=0.05,
                    label="Confidence threshold",
                )
                run_button = gr.Button("Run detection")

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

    return app


if __name__ == "__main__":
    build_app().launch()
