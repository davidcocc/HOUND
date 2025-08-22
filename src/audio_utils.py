import numpy as np
import librosa


def extract_mel_spectrogram(file_path: str, fixed_length: int = 168):
    """
    Extract mel spectrogram from an audio file.
    
    Args:
        file_path: Path to the audio file
        fixed_length: Fixed length for the spectrogram (default: 168)
        
    Returns:
        Tuple of (mel_spectrogram, sample_rate, audio_data)
        Returns (None, None, None) if processing fails
    """
    try:
        audio, sr = librosa.load(file_path, sr=22050, mono=True)
        n_fft = min(2048, len(audio))
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, n_mels=168, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] > fixed_length:
            mel_db = mel_db[:, :fixed_length]
        else:
            mel_db = np.pad(mel_db, ((0, 0), (0, fixed_length - mel_db.shape[1])), mode="constant")

        return mel_db, sr, audio
    except Exception as exc:
        print(f"❌ Error processing {file_path}: {exc}")
        return None, None, None

def audio_augmentation(audio, sample_rate):
    augmentations = []

    # Add noise
    noise = np.random.normal(0, 0.005, audio.shape)
    augmented_audio = audio + noise
    augmentations.append(augmented_audio)

    # Add pitch shift
    pitch_shift = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=np.random.randint(-2, 2))
    augmentations.append(pitch_shift)

    # Add time stretch
    time_stretch = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    augmentations.append(time_stretch)

    return augmentations


def extract_mel_spectrogram_from_audio(audio, sample_rate, fixed_length=168):
    """
    Extract mel spectrogram from audio data.
    
    Args:
        audio: Audio data array
        sample_rate: Sample rate of the audio
        fixed_length: Fixed length for the spectrogram (default: 168)
        
    Returns:
        Mel spectrogram array or None if processing fails
    """
    try:
        n_fft = min(2048, len(audio))
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, n_mels=168, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] > fixed_length:
            mel_db = mel_db[:, :fixed_length]
        else:
            mel_db = np.pad(mel_db, ((0, 0), (0, fixed_length - mel_db.shape[1])), mode="constant")

        return mel_db
    except Exception as exc:
        print(f"❌ Error processing audio data: {exc}")
        return None