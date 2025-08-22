# Audio Classification Utilities

Questo package fornisce utilità per la classificazione audio, organizzate in moduli separati per una migliore manutenibilità.

## Struttura del Package

### `utils.py`
Funzioni di utilità generale:
- `resolve_project_root()`: Risolve il percorso root del progetto

### `audio_utils.py`
Funzioni per la manipolazione e l'elaborazione audio:
- `extract_mel_spectrogram()`: Estrae mel spectrogram da file audio

### `model_utils.py`
Funzioni relative al modello di machine learning:
- `load_keras_model()`: Carica un modello Keras
- `load_metadata()`: Carica i metadati dal CSV
- `predict_from_features()`: Effettua predizioni da features
- `predict_file()`: Predice la classe di un file audio
- `evaluate_model()`: Valuta le performance del modello

## Utilizzo

```python
# Import delle funzioni
from utils import resolve_project_root
from audio_utils import extract_mel_spectrogram
from model_utils import load_keras_model, predict_file

# Esempio di utilizzo
project_root = resolve_project_root(__file__)
model = load_keras_model(project_root)
cls, probs, sr, audio = predict_file(model, "path/to/audio.wav")
```

## Dipendenze

- `numpy`: Per operazioni numeriche
- `librosa`: Per elaborazione audio
- `tensorflow`: Per il modello di machine learning
- `pandas`: Per gestione dati
- `matplotlib`, `seaborn`: Per visualizzazioni
- `scikit-learn`: Per metriche di valutazione
