"""Run the Gradio inference application."""

from __future__ import annotations

from pothole_severity_detection.apps.gradio_app import build_app


def main() -> None:
    """Launch the Gradio app."""
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
