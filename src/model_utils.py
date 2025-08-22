import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import seaborn as sns

from audio_utils import extract_mel_spectrogram, audio_augmentation, extract_mel_spectrogram_from_audio


def ensure_metrics_directory():
    """Ensure the metrics directory exists."""
    metrics_dir = "metrics"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
        print(f"✅ Created metrics directory: {metrics_dir}")


def load_keras_model(project_root: str, model_rel_path: str = os.path.join("model", "UrbanSound8K.keras")):
    """
    Load a Keras model from the specified path.
    
    Args:
        project_root: Root directory of the project
        model_rel_path: Relative path to the model file
        
    Returns:
        Loaded Keras model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    from keras.models import load_model  # Local import to avoid hard dependency during docs or light tools

    model_path = os.path.join(project_root, model_rel_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at relative path: {model_path}")
    return load_model(model_path)


def load_metadata(project_root: str, csv_rel_path: str = os.path.join("data", "UrbanSound8K.csv")):
    """
    Load metadata from CSV file and create class mapping.
    
    Args:
        project_root: Root directory of the project
        csv_rel_path: Relative path to the metadata CSV file
        
    Returns:
        Tuple of (metadata DataFrame, class_mapping dictionary)
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    metadata_path = os.path.join(project_root, csv_rel_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata CSV not found at relative path: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    class_mapping = dict(zip(metadata["classID"], metadata["class"]))
    return metadata, class_mapping


@tf.function(reduce_retracing=True)
def _tf_infer(model, input_tensor: tf.Tensor):
    """
    TensorFlow function for model inference.
    
    Args:
        model: Keras model
        input_tensor: Input tensor for prediction
        
    Returns:
        Model prediction
    """
    return model(input_tensor, training=False)


def predict_from_features(model, mel_features: np.ndarray):
    """
    Make prediction from mel spectrogram features.
    
    Args:
        model: Keras model
        mel_features: Mel spectrogram features
        
    Returns:
        Tuple of (predicted_class, probabilities)
    """
    features = (mel_features - np.mean(mel_features)) / np.std(mel_features)
    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=0)
    input_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    prediction = _tf_infer(model, input_tensor)
    probs = prediction.numpy()[0]
    predicted_class = int(np.argmax(probs))
    return predicted_class, probs


def predict_file(model, file_path: str):
    """
    Predict class for an audio file.
    
    Args:
        model: Keras model
        file_path: Path to the audio file
        
    Returns:
        Tuple of (predicted_class, probabilities, sample_rate, audio_data)
        Returns (None, None, None, None) if processing fails
    """
    mel, sr, audio = extract_mel_spectrogram(file_path)
    if mel is None:
        return None, None, None, None
    cls, probs = predict_from_features(model, mel)
    return cls, probs, sr, audio


def evaluate_model(model, test_files, test_labels, use_augmentation=False):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Keras model
        test_files: List of test file paths
        test_labels: List of true labels
        use_augmentation: Whether to apply audio augmentation
        
    Returns:
        Tuple of (y_true, y_pred) lists
    """
    y_true, y_pred = [], []
    y_prob_list = []

    for file_path, label in zip(test_files, test_labels):
        mel, sr, audio = extract_mel_spectrogram(file_path)
        if mel is None:
            continue

        if use_augmentation and audio is not None:
            # Apply augmentation and use the first augmented version
            augmented_audios = audio_augmentation(audio, sr)
            if augmented_audios:
                # Use the first augmented version (noise)
                augmented_audio = augmented_audios[0]
                # Re-extract features from augmented audio
                mel = extract_mel_spectrogram_from_audio(augmented_audio, sr)
                if mel is None:
                    continue

        predicted_class, probs = predict_from_features(model, mel)
        y_true.append(label)
        y_pred.append(predicted_class)
        y_prob_list.append(probs)

        print(f"Predicted: {predicted_class}, True: {label}")

    # Ensure metrics directory exists
    ensure_metrics_directory()
    
    # Determine suffix for file names
    suffix = "_augmented" if use_augmentation else "_normal"
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix{suffix}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"metrics/confusion_matrix{suffix}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Save metrics to file
    metrics_text = f"Accuracy: {acc:.4f}\n"
    metrics_text += f"Precision (weighted): {prec:.4f}\n"
    metrics_text += f"Recall (weighted): {rec:.4f}\n"
    metrics_text += f"F1-score (weighted): {f1:.4f}\n"
    
    with open(f"metrics/metrics{suffix}.txt", "w") as f:
        f.write(metrics_text)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")

    # ROC Curves (one-vs-rest)
    if y_prob_list:
        y_score = np.array(y_prob_list)
        num_classes = y_score.shape[1]
        classes = list(range(num_classes))
        y_true_bin = label_binarize(y_true, classes=classes)

        plt.figure(figsize=(10, 8))
        for class_idx in classes:
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_score[:, class_idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label=f"Class {class_idx} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves (One-vs-Rest){suffix}")
        plt.legend(loc="lower right", fontsize="small")
        plt.savefig(f"metrics/roc_curve{suffix}.png", dpi=300, bbox_inches='tight')
        plt.show()

    return y_true, y_pred
