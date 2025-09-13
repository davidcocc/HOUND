import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from audio_utils import extract_mel_spectrogram
from utils import resolve_project_root
import time

NUM_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
CONV_FILTERS = [32, 64, 128, 256]

def create_model(input_shape=(168, 168, 1), num_classes=10):
    model = Sequential()
    
    model.add(Conv2D(CONV_FILTERS[0], (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(CONV_FILTERS[1], (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(CONV_FILTERS[2], (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(CONV_FILTERS[3], (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    return model

def load_data(project_root):
    metadata_path = os.path.join(project_root, 'data', 'UrbanSound8K.csv')
    metadata = pd.read_csv(metadata_path)
    
    X, y = [], []
    for _, row in metadata.iterrows():
        if row['fold'] == 10: continue
        file_path = os.path.join(project_root, 'data', 'archive', f"fold{row['fold']}", row['slice_file_name'])
        mel, _, _ = extract_mel_spectrogram(file_path)
        if mel is not None:
            X.append(mel)
            y.append(row['classID'])
    
    X = np.array(X)[..., np.newaxis]
    y = to_categorical(LabelEncoder().fit_transform(y), num_classes=10)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == '__main__':
    root = resolve_project_root(__file__)
    X_train, X_val, y_train, y_val = load_data(root)
    
    model = create_model()
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    training_time = time.time() - start_time
    
    print(f"✅ Tempo di addestramento: {training_time:.2f} secondi")
    print(f"Accuracy finale di training: {history.history['accuracy'][-1]:.4f}")
    print(f"Accuracy finale di validation: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Loss finale di training: {history.history['loss'][-1]:.4f}")
    print(f"Loss finale di validation: {history.history['val_loss'][-1]:.4f}")
    
    custom_model_path = os.path.join(root, 'custom_model', 'custom_UrbanSound8K.keras')
    model.save(custom_model_path)
    print(f"✅ Modello custom salvato in: {custom_model_path}")
