import os
import argparse
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import resolve_project_root
from model_utils import (
    load_keras_model,
    load_metadata,
    predict_from_features,
    predict_file,
    evaluate_model,
)
from audio_utils import extract_mel_spectrogram, audio_augmentation, export_augmented_versions

try:
    from IPython.display import Audio, display
except Exception:
    Audio = None
    display = None

_HERE = os.path.dirname(os.path.abspath(__file__))
# Project root (one level up from src/)
_ROOT = resolve_project_root(__file__)
model = load_keras_model(_ROOT)


# ✅ Load metadata & create class mapping
metadata, class_mapping = load_metadata(_ROOT)

# ✅ Feature Extraction Function
def extract_features(file_path, fixed_length=168):
    return extract_mel_spectrogram(file_path, fixed_length=fixed_length)

# ✅ Use @tf.function to prevent excessive retracing
@tf.function(reduce_retracing=True)
def make_prediction(model, input_tensor):
    return model(input_tensor, training=False)

def predict_audio(file_path):
    """Loads an audio file, extracts features, and predicts its class."""
    cls, probs, sr, audio = predict_file(model, file_path)
    if cls is None:
        print("❌ Error extracting features.")
        return

    predicted_label = class_mapping.get(cls, "Unknown")
    print(f"✅ Predicted Class: {predicted_label} (ID: {cls})")

    if Audio and display and audio is not None and sr is not None:
        display(Audio(audio, rate=sr))

    mel, _, _ = extract_features(file_path)
    if mel is not None:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Mel Spectrogram - {predicted_label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="UrbanSound8K inference utility")
    parser.add_argument("--file", "-f", type=str, default=os.path.join(_ROOT, "data", "archive", "fold2", "166421-3-0-5.wav"), help="Path to an input .wav file")
    args = parser.parse_args()
    if not os.path.exists(args.file):
        print(f"❌ Provided file does not exist: {args.file}")
        return
    predict_audio(args.file)

    # Export augmented versions
    output_dir = os.path.join(_ROOT, "augmented")
    export_augmented_versions(args.file, output_dir)

    test_meta = metadata[metadata["fold"] == 2]
    test_files = [
        os.path.join(_ROOT, "data", "archive", f"fold{fold}", fname)
        for fold, fname in zip(test_meta["fold"], test_meta["slice_file_name"])
    ]
    test_labels = test_meta["classID"].tolist()

    # Esegui valutazione sui dati normali
    #print("🔍 Valutazione modello sui dati normali...")
    # evaluate_model(model, test_files, test_labels)
    
    # Esegui valutazione sui dati augmentati
    #print("🔍 Valutazione modello sui dati augmentati...")
    #evaluate_model(model, test_files, test_labels, use_augmentation=True)

if __name__ == "__main__":
    main()