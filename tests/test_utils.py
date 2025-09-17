"""Unit tests for src.utils plotting helpers and default selection logic."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_generate_probabilities_pie_normalizes_and_returns_figure():
    from src import utils

    probs = {"a": 0.0, "b": 0.0, "c": 0.0}
    fig = utils.generate_probabilities_pie(probs)
    assert isinstance(fig, plt.Figure)

    probs2 = {"a": 0.2, "b": 0.3, "c": 0.5}
    fig2 = utils.generate_probabilities_pie(probs2)
    assert isinstance(fig2, plt.Figure)


def test_generate_mel_spectrogram_figure_handles_missing(tmp_path):
    from src import utils

    missing = tmp_path / "no.wav"
    fig = utils.generate_mel_spectrogram_figure(str(missing))
    assert fig is None


def test_choose_default_model_label_prefers_custom():
    from src import utils

    options = {
        "Original model": "model/UrbanSound8K.keras",
        "Custom: foo": "custom_model/foo.keras",
    }
    assert utils.choose_default_model_label(options).startswith("Custom")

