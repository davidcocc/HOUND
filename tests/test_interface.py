"""Unit tests for src.interface classify logic.

These tests validate the pure Python functions without starting the Gradio UI.
We mock heavy I/O and model loading where appropriate.
"""

import os
from typing import Dict
from unittest import mock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _patch_root_tmpdir(tmp_path, monkeypatch):
    """Ensure interface resolves models and metadata in a temporary structure.

    We create a fake project layout with:
    - data/UrbanSound8K.csv (minimal mapping)
    - model/UrbanSound8K.keras (empty sentinel file)
    """
    # Build fake project tree
    root = tmp_path
    data_dir = root / "data"
    model_dir = root / "model"
    custom_dir = root / "custom_model"
    data_dir.mkdir()
    model_dir.mkdir()
    custom_dir.mkdir()

    # Minimal CSV for class mapping
    (data_dir / "UrbanSound8K.csv").write_text("classID,class\n0,air_conditioner\n1,car_horn\n")
    # Sentinel keras file
    (model_dir / "UrbanSound8K.keras").write_text("sentinel")

    # Patch interface module constants to point to tmp root
    import importlib

    interface = importlib.import_module("src.interface")
    monkeypatch.setattr(interface, "_ROOT", str(root), raising=True)
    # Invalidate LRU caches to ensure they re-read structure
    interface.get_model_options.cache_clear()
    interface._get_loaded_model.cache_clear()
    interface._get_class_mapping.cache_clear()

    yield


def test_discover_models_and_classify_success(monkeypatch):
    import importlib

    interface = importlib.import_module("src.interface")

    # Fake audio file path
    fake_audio = os.path.join(interface._ROOT, "fake.wav")
    open(fake_audio, "w").close()

    # Ensure a model option exists
    options = interface.get_model_options()
    assert options, "Expected at least one model option"
    label = next(iter(options.keys()))

    # Mock model loading and prediction
    def _dummy_model_loader(_root, rel):  # noqa: ARG001
        class DummyModel:
            pass

        return DummyModel()

    def _dummy_predict_file(model, path):  # noqa: ARG001
        probs = np.array([0.2, 0.8], dtype=float)
        return 1, probs, 22050, np.zeros(10)

    monkeypatch.setattr(interface, "load_keras_model", _dummy_model_loader, raising=True)
    monkeypatch.setattr(interface, "predict_file", _dummy_predict_file, raising=True)
    interface._get_loaded_model.cache_clear()

    message, probs = interface.classify(fake_audio, label)

    assert "Predicted:" in message
    # Probabilities normalized
    assert pytest.approx(sum(probs.values()), rel=1e-6) == 1.0
    # The predicted label should correspond to classID 1 -> car_horn
    assert any("car_horn" in message for message in [message])
    assert set(probs.keys()) == {"air_conditioner", "car_horn"}


def test_classify_handles_missing_file():
    import importlib

    interface = importlib.import_module("src.interface")
    msg, probs = interface.classify("/no/such.wav", "any")
    assert "No valid audio file" in msg
    assert probs == {}


def test_classify_handles_model_load_failure(monkeypatch):
    import importlib

    interface = importlib.import_module("src.interface")

    # Provide a real file path but broken loader
    fake_audio = os.path.join(interface._ROOT, "fake2.wav")
    open(fake_audio, "w").close()

    options = interface.get_model_options()
    label = next(iter(options.keys())) if options else "Original model"

    def _broken_loader(_root, rel):  # noqa: ARG001
        raise FileNotFoundError("missing")

    monkeypatch.setattr(interface, "load_keras_model", _broken_loader, raising=True)
    interface._get_loaded_model.cache_clear()

    msg, probs = interface.classify(fake_audio, label)
    assert "Model file not found" in msg
    assert probs == {}


def test_classify_handles_predict_failure(monkeypatch):
    import importlib

    interface = importlib.import_module("src.interface")

    fake_audio = os.path.join(interface._ROOT, "fake3.wav")
    open(fake_audio, "w").close()

    options = interface.get_model_options()
    label = next(iter(options.keys()))

    def _dummy_loader(_root, rel):  # noqa: ARG001
        class DummyModel:
            pass

        return DummyModel()

    def _bad_predict_file(model, path):  # noqa: ARG001
        return None, None, None, None

    monkeypatch.setattr(interface, "load_keras_model", _dummy_loader, raising=True)
    monkeypatch.setattr(interface, "predict_file", _bad_predict_file, raising=True)
    interface._get_loaded_model.cache_clear()

    msg, probs = interface.classify(fake_audio, label)
    assert "Failed to process the audio" in msg
    assert probs == {}


