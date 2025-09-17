import os
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from model_utils import (
    load_keras_model,
    load_metadata,
    predict_file,
    evaluate_custom_vs_original,
)
from audio_utils import extract_mel_spectrogram

try:
    from IPython.display import Audio, display
except Exception:
    Audio = None
    display = None

_HERE = os.path.dirname(os.path.abspath(__file__))
# Project root (one level up from src/)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
# model = load_keras_model(_ROOT)
model = load_keras_model(_ROOT, os.path.join("custom_model", "custom_UrbanSound8K.keras"))


# Load metadata & create class mapping
metadata, class_mapping = load_metadata(_ROOT)

# Feature Extraction Function
def extract_features(file_path, fixed_length=168):
    return extract_mel_spectrogram(file_path, fixed_length=fixed_length)

# Use @tf.function to prevent excessive retracing
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
    parser.add_argument("--file", "-f", type=str, default=os.path.join(_ROOT, "data", "archive", "testsounds", "254779__jonathantremblay__air-conditioner.wav"), help="Path to an input .wav file")
    parser.add_argument("--compare", action="store_true", help="Evaluate and compare original vs custom models on fold 10")
    parser.add_argument("--original-model-rel", type=str, default=os.path.join("model", "UrbanSound8K.keras"), help="Relative path to original model from project root")
    parser.add_argument("--custom-model-rel", type=str, default=os.path.join("custom_model", "custom_UrbanSound8K.keras"), help="Relative path to custom model from project root")
    args = parser.parse_args()
    
    if args.compare:
        results = evaluate_custom_vs_original(
            _ROOT,
            original_rel_path=args.original_model_rel,
            custom_rel_path=args.custom_model_rel,
        )
        print(f"Confronto accuracies -> Originale: {results['original_accuracy']:.4f} | Custom: {results['custom_accuracy']:.4f}")
        return
    if not os.path.exists(args.file):
        print(f"❌ Provided file does not exist: {args.file}")
        return
    predict_audio(args.file)

if __name__ == "__main__":
    main()