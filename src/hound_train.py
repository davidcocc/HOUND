import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from audio_utils import extract_mel_spectrogram, audio_augmentation, extract_mel_spectrogram_from_audio
from keras.metrics import Precision, Recall
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import time
import GPUtil
import tensorflow as tf

NUM_EPOCHS = 90
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
CONV_FILTERS = [32, 64, 128, 256]

def create_model(input_shape=(168, 168, 1), num_classes=10):
    model = Sequential()
    
    model.add(Conv2D(CONV_FILTERS[0], (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(CONV_FILTERS[1], (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))  
    
    model.add(Conv2D(CONV_FILTERS[2], (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3)) 
    
    model.add(Conv2D(CONV_FILTERS[3], (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))) 
    model.add(Dropout(0.6))  
    model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))  
    model.add(Dropout(0.6))  
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy', Precision(), Recall()]
    )
    return model

def load_data(project_root, use_augmentation=True):
    metadata_path = os.path.join(project_root, 'data', 'UrbanSound8K.csv')
    metadata = pd.read_csv(metadata_path)

    # Prepare containers for folds: train (1-8), val (9), test (10)
    X_train_raw, y_train_raw, audio_train, sr_train = [], [], [], []
    X_val_raw, y_val_raw, audio_val, sr_val = [], [], [], []
    X_test_raw, y_test_raw = [], []

    for _, row in metadata.iterrows():
        fold = int(row['fold'])
        file_path = os.path.join(project_root, 'data', 'archive', f"fold{fold}", row['slice_file_name'])
        mel, sr, audio = extract_mel_spectrogram(file_path)
        if mel is None:
            continue
        if 1 <= fold <= 8:
            X_train_raw.append(mel)
            y_train_raw.append(row['class'])
            audio_train.append(audio)
            sr_train.append(sr)
        elif fold == 9:
            X_val_raw.append(mel)
            y_val_raw.append(row['class'])
            audio_val.append(audio)
            sr_val.append(sr)
        elif fold == 10:
            X_test_raw.append(mel)
            y_test_raw.append(row['class'])

    # Fit label encoder on full class set to ensure consistent mapping
    encoder = LabelEncoder()
    encoder.fit(metadata['class'])

    # One-hot encode labels
    num_classes = len(encoder.classes_)
    y_train = to_categorical(encoder.transform(y_train_raw), num_classes=num_classes)
    y_val = to_categorical(encoder.transform(y_val_raw), num_classes=num_classes)
    y_test = to_categorical(encoder.transform(y_test_raw), num_classes=num_classes)

    # Add channel dimension
    X_train = np.array(X_train_raw)[..., np.newaxis]
    X_val = np.array(X_val_raw)[..., np.newaxis]
    X_test = np.array(X_test_raw)[..., np.newaxis]

    # Augment only training data if flag is True
    X_train_list = list(X_train)
    y_train_list = list(y_train)
    if use_augmentation:
        for i in range(len(audio_train)):
            augmentations = audio_augmentation(audio_train[i], sr_train[i])
            for aug_audio in augmentations:
                aug_mel = extract_mel_spectrogram_from_audio(aug_audio, sr_train[i])
                if aug_mel is not None:
                    X_train_list.append(aug_mel[..., np.newaxis])
                    y_train_list.append(y_train[i])

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    # Dataset normalization using train mean/std
    eps = 1e-8
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    X_train = (X_train - train_mean) / (train_std + eps)
    X_val = (X_val - train_mean) / (train_std + eps)
    X_test = (X_test - train_mean) / (train_std + eps)

    return X_train, X_val, X_test, y_train, y_val, y_test, encoder

if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    X_train, X_val, X_test, y_train, y_val, y_test, encoder = load_data(root, use_augmentation=True)  # Set to False to disable augmentation
    
    model = create_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=12, verbose=1, restore_best_weights=True)
    checkpoint_path = os.path.join(root, 'custom_model', 'best_custom_UrbanSound8K.keras')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001, verbose=1)

    # Track resource usage
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=None)
    gpus = GPUtil.getGPUs()
    start_gpu_load = gpus[0].load * 100 if gpus else 0
    start_gpu_mem = gpus[0].memoryUtil * 100 if gpus else 0
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkpoint, lr_scheduler]
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    end_cpu = psutil.cpu_percent(interval=None)
    avg_cpu = (start_cpu + end_cpu) / 2
    
    gpus = GPUtil.getGPUs()
    end_gpu_load = gpus[0].load * 100 if gpus else 0
    end_gpu_mem = gpus[0].memoryUtil * 100 if gpus else 0
    avg_gpu_load = (start_gpu_load + end_gpu_load) / 2
    avg_gpu_mem = (start_gpu_mem + end_gpu_mem) / 2
    
    # Estimated energy consumption (simplified: time * avg power estimate; adjust power values as needed)
    cpu_power_estimate = 50  # Watts, approximate for CPU
    gpu_power_estimate = 150 if gpus else 0  # Watts, approximate for GPU
    estimated_energy = (training_time / 3600) * (avg_cpu / 100 * cpu_power_estimate + avg_gpu_load / 100 * gpu_power_estimate)  # in Wh
    
    print(f"✅ Tempo di addestramento: {training_time:.2f} secondi")
    print(f"Accuracy finale di training: {history.history['accuracy'][-1]:.4f}")
    print(f"Accuracy finale di validation: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Loss finale di training: {history.history['loss'][-1]:.4f}")
    print(f"Loss finale di validation: {history.history['val_loss'][-1]:.4f}")
    
    # Evaluation on validation set
    y_pred_val = model.predict(X_val)
    y_pred_val_classes = np.argmax(y_pred_val, axis=1)
    y_true_val_classes = np.argmax(y_val, axis=1)
    class_report_val = classification_report(y_true_val_classes, y_pred_val_classes, target_names=encoder.classes_)
    print(class_report_val)

    # Confusion matrix (validation)
    cm_val = confusion_matrix(y_true_val_classes, y_pred_val_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix - Validation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    confusion_val_path = os.path.join(root, 'metrics', 'custom', 'confusion_matrix_custom_val.png')
    os.makedirs(os.path.dirname(confusion_val_path), exist_ok=True)
    plt.savefig(confusion_val_path)
    plt.close()
    print(f"✅ Matrice di confusione (val) salvata in: {confusion_val_path}")

    # Evaluation on test set
    y_pred_test = model.predict(X_test)
    y_pred_test_classes = np.argmax(y_pred_test, axis=1)
    y_true_test_classes = np.argmax(y_test, axis=1)
    class_report_test = classification_report(y_true_test_classes, y_pred_test_classes, target_names=encoder.classes_)
    print(class_report_test)

    # Confusion matrix (test)
    cm_test = confusion_matrix(y_true_test_classes, y_pred_test_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix - Test')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    confusion_test_path = os.path.join(root, 'metrics', 'custom', 'confusion_matrix_custom_test.png')
    os.makedirs(os.path.dirname(confusion_test_path), exist_ok=True)
    plt.savefig(confusion_test_path)
    plt.close()
    print(f"✅ Matrice di confusione (test) salvata in: {confusion_test_path}")

    # Export all metrics to files (validation and test)
    metrics_val_path = os.path.join(root, 'metrics', 'custom', 'metrics_custom_val.txt')
    metrics_test_path = os.path.join(root, 'metrics', 'custom', 'metrics_custom_test.txt')
    os.makedirs(os.path.dirname(metrics_val_path), exist_ok=True)
    with open(metrics_val_path, 'w') as f:
        f.write(f"Tempo di addestramento: {training_time:.2f} secondi\n")
        f.write(f"Accuracy finale di training: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Accuracy finale di validation: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Loss finale di training: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Loss finale di validation: {history.history['val_loss'][-1]:.4f}\n")
        f.write("\nClassification Report (Validation):\n")
        f.write(class_report_val)
        f.write("\nResource Usage:\n")
        f.write(f"Average CPU Usage: {avg_cpu:.2f}%\n")
        f.write(f"Average GPU Load: {avg_gpu_load:.2f}%\n")
        f.write(f"Average GPU Memory: {avg_gpu_mem:.2f}%\n")
        f.write(f"Estimated Energy Consumption: {estimated_energy:.2f} Wh\n")
        f.write(f"\nMatrice di confusione (val) salvata in: {confusion_val_path}\n")
    print(f"✅ Metriche (validation) esportate in: {metrics_val_path}")

    with open(metrics_test_path, 'w') as f:
        f.write("Classification Report (Test):\n")
        f.write(class_report_test)
        f.write(f"\nMatrice di confusione (test) salvata in: {confusion_test_path}\n")
    print(f"✅ Metriche (test) esportate in: {metrics_test_path}")

    custom_model_path = os.path.join(root, 'custom_model', 'custom_UrbanSound8K.keras')
    model.save(custom_model_path)
    print(f"✅ Modello custom salvato in: {custom_model_path}")
