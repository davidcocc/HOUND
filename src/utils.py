"""Utility helpers for visualization and simple UI logic.

This module contains plotting helpers used by the Gradio interface and simple
pure functions that encapsulate UI-related logic. Keeping these helpers here
allows unit tests to validate behavior without exercising the Gradio runtime.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    # Preferred when used as part of the package
    from .audio_utils import extract_mel_spectrogram
except Exception:
    # Fallback when running as a script without package context
    from audio_utils import extract_mel_spectrogram


def generate_mel_spectrogram_figure(file_path: str) -> Optional[plt.Figure]:
    """Generate a matplotlib Figure showing a mel spectrogram for an audio file.

    Returns None if the audio cannot be processed.
    """
    mel_db, sample_rate, _audio = extract_mel_spectrogram(file_path)
    if mel_db is None or sample_rate is None:
        return None

    # Non-trivial plotting: use specshow-like image with time/frequency axes.
    fig, ax = plt.subplots(figsize=(8, 3))
    # Display mel spectrogram as dB image; rely on imshow for minimal deps
    im = ax.imshow(mel_db, aspect="auto", origin="lower")
    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel bins")
    fig.colorbar(im, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def generate_probabilities_pie(probabilities: Dict[str, float]) -> plt.Figure:
    """Create a pie chart Figure from a mapping of class label -> probability.

    The function normalizes probabilities defensively so the chart is always
    rendered, even with unnormalized inputs.
    """
    labels = list(probabilities.keys())
    values = np.array(list(probabilities.values()), dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        # Avoid division by zero; render uniform distribution as a fallback
        values = np.ones_like(values)
        total = float(values.sum())
    values = values / total

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct="%1.1f%%")
    ax.set_title("Class Probabilities")
    fig.tight_layout()
    return fig


def choose_default_model_label(options: Dict[str, str]) -> str:
    """Choose default model label, preferring custom models when available.

    The `options` map contains UI label -> relative model path. We look for any
    option whose path begins with 'custom_model' or whose label starts with 'Custom'.
    If none is found, we return the first available label.
    """
    if not options:
        return ""
    # Prefer custom models by path or label
    for label, rel in options.items():
        if rel.replace("\\", "/").startswith("custom_model/") or label.lower().startswith("custom"):
            return label
    # Fallback to first
    return next(iter(options.keys()))


