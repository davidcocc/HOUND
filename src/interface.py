"""Gradio interface for UrbanSound8K audio classification.

This module exposes a minimal Gradio UI to:
- upload or record audio
- select a model from available .keras files
- display the predicted class and per-class probabilities

It reuses the project's existing utilities from `model_utils` and `audio_utils`.

The UI is intentionally simple and avoids overengineering, following the
project's Cursor rules and PEP8 guidelines.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import gradio as gr

try:
    from .model_utils import load_keras_model, load_metadata, predict_file
except Exception:  # Allow running as script: python -m src.interface
    from model_utils import load_keras_model, load_metadata, predict_file


# Resolve project root (one level up from src/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))


def _is_keras_file(path: str) -> bool:
    """Return True if the path points to a Keras model file."""
    return path.endswith(".keras") and os.path.isfile(path)


def _discover_model_paths(project_root: str) -> Dict[str, str]:
    """Discover available .keras models under known directories.

    Returns a mapping of human-friendly option labels to relative paths.
    """
    candidates: List[Tuple[str, str]] = []

    # Default/original model
    default_rel = os.path.join("model", "UrbanSound8K.keras")
    default_abs = os.path.join(project_root, default_rel)
    if _is_keras_file(default_abs):
        candidates.append(("Original model", default_rel))

    # Custom models (collect all .keras files in custom_model/)
    custom_dir = os.path.join(project_root, "custom_model")
    if os.path.isdir(custom_dir):
        for fname in sorted(os.listdir(custom_dir)):
            if fname.endswith(".keras"):
                rel = os.path.join("custom_model", fname)
                label = f"Custom: {os.path.splitext(fname)[0]}"
                if _is_keras_file(os.path.join(project_root, rel)):
                    candidates.append((label, rel))

    # Fallback to empty dict if nothing found
    return {label: rel for label, rel in candidates}


@lru_cache(maxsize=1)
def get_model_options() -> Dict[str, str]:
    """Get a cached mapping of dropdown label -> relative model path."""
    return _discover_model_paths(_ROOT)


@lru_cache(maxsize=4)
def _get_loaded_model(model_rel_path: str):
    """Load and cache a Keras model by its relative path from project root."""
    return load_keras_model(_ROOT, model_rel_path)


@lru_cache(maxsize=1)
def _get_class_mapping() -> Dict[int, str]:
    """Load and cache the UrbanSound8K classID->class label mapping."""
    _, class_mapping = load_metadata(_ROOT)
    return class_mapping


def classify(filepath: Optional[str], model_label: str) -> Tuple[str, Dict[str, float]]:
    """Classify an audio file with the selected model.

    Args:
        filepath: Path to the audio file on disk. If None or not found, returns an error.
        model_label: UI label of the selected model option.

    Returns:
        Tuple of (message string, probabilities dict by class label).
        On error, message contains an explanation and the probabilities dict is empty.
    """
    if not filepath or not os.path.exists(filepath):
        return ("No valid audio file provided.", {})

    options = get_model_options()
    if not options:
        return ("No models found in the repository.", {})

    rel_path = options.get(model_label)
    if rel_path is None:
        # Fall back to first available model if selection is stale
        first_label = next(iter(options.keys()))
        rel_path = options[first_label]
        model_label = first_label

    try:
        model = _get_loaded_model(rel_path)
    except FileNotFoundError:
        return (f"Model file not found: {rel_path}", {})
    except Exception as exc:  # Non-trivial: loading can fail for multiple reasons
        return (f"Failed to load model: {exc}", {})

    class_mapping = _get_class_mapping()

    cls, probs, _sr, _audio = predict_file(model, filepath)
    if cls is None or probs is None:
        return ("Failed to process the audio. Please try another file.", {})

    # Map to human-readable labels
    id_to_label = class_mapping
    label_probs: Dict[str, float] = {}

    # Normalize to sum=1.0 if needed; keep it simple and robust.
    total = float(sum(probs)) if probs is not None else 0.0
    for class_id, p in enumerate(probs):
        label = id_to_label.get(class_id, f"Class {class_id}")
        label_probs[label] = float(p if total <= 0.0 else p / total)

    predicted_label = id_to_label.get(int(cls), f"Class {int(cls)}")
    message = f"Predicted: {predicted_label}  |  Model: {model_label}"
    return message, label_probs


def build_interface() -> gr.Blocks:
    """Build and return the Gradio Blocks app using a dark, accessible theme."""
    options = get_model_options()
    default_choice = next(iter(options.keys())) if options else ""

    theme = gr.themes.Soft()

    with gr.Blocks(theme=theme, title="UrbanSound Classifier") as demo:
        gr.Markdown(
            """
            ### UrbanSound8K Classifier
            Upload or record an audio clip, choose a model, and run classification.
            """.strip()
        )

        with gr.Row():
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Audio (upload or record)",
            )
            model_dd = gr.Dropdown(
                choices=list(options.keys()) or [""],
                value=default_choice,
                label="Model",
            )

        run_btn = gr.Button("Classify", variant="primary")

        with gr.Row():
            pred_txt = gr.Textbox(label="Prediction", interactive=False)
            prob_json = gr.JSON(label="Probabilities by class")

        def _on_click(file_path: Optional[str], model_choice: str):
            msg, probs = classify(file_path, model_choice)
            return msg, probs

        run_btn.click(_on_click, inputs=[audio_in, model_dd], outputs=[pred_txt, prob_json])

        gr.Markdown(
            """
            Note: This demo is optimized for clarity. Results may vary depending on the
            selected model and the audio content.
            """.strip()
        )

    return demo


def main() -> None:
    """Launch the Gradio app."""
    demo = build_interface()
    # Keep queue and defaults simple; avoid overengineering server options
    demo.queue().launch()


if __name__ == "__main__":
    main()


