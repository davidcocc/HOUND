# HOUND: High-fidelity Optimized Urban Noise Detection

[![CI/CD Pipeline](https://github.com/davidcocc/HOUND/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/davidcocc/HOUND/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

HOUND is an audio classification system optimized for urban noise detection, built on the UrbanSound8K dataset. It features a custom deep learning model for high-fidelity sound classification, a user-friendly Gradio interface for inference, and robust CI/CD pipelines.

This project was developed as part of the Software Engineering for Artificial Intelligence course at University of Salerno.

## Authors
- [Coccorullo David](https://github.com/davidcocc)
- [Zunico Anthony](https://github.com/DJHeisenberg01)

## Features
- Custom CNN model for urban sound classification.
- Data augmentation and mel spectrogram extraction for improved accuracy.
- Gradio-based web interface for easy inference.
- Docker support for containerization.
- Comprehensive unit tests with pytest and coverage reports.
- CI/CD workflow with linting, security scans, and artifact uploads.

## Installation
### Prerequisites
- Python 3.10+
- pip for package management

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hound.git
   cd hound
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: For testing, additionally install:
   ```bash
   pip install pytest pytest-cov pytest-html gradio
   ```

3. (Optional) Set up Docker:
   - Build the image: `docker build -t hound .`
   - Run: `docker run -p 7860:7860 hound`

## Usage
### Training
To train or retrain the custom model:
```bash
python -m src.hound_train --dataset data/archive/ --output custom_model/custom_UrbanSound8K.keras
```
- Use `--augment` for data augmentation.
- Metrics and visualizations are saved in `metrics/custom/`.

### Inference
Run inference on a single audio file:
```bash
python -m src.hound_inference --file path/to/audio.wav
```
- Use `--compare` to evaluate original vs. custom model on fold 10.

### Gradio Interface
Launch the web UI:
```bash
python -m src.interface
```
- Upload an audio file.
- Select a model via buttons (defaults to custom).
- Click "Classify" to see prediction, spectrogram, and probability pie chart.

### Testing
Run unit tests with coverage and reports:
```bash
pytest --cov=src --cov-report=html --html=report.html
```
- View `htmlcov/index.html` for coverage.
- View `report.html` for test results.

## Model Card
### Model Overview
- **Name**: Custom UrbanSound8K CNN
- **Version**: 1.0
- **Description**: A convolutional neural network fine-tuned on the UrbanSound8K dataset for classifying 10 urban sound classes (e.g., air_conditioner, car_horn).
- **Architecture**: CNN with mel spectrogram inputs (168x168), trained with data augmentation (noise, pitch shift, time stretch).
- **Training Data**: UrbanSound8K (8732 labeled sound excerpts ≤4s across 10 folds).
- **Performance**:
  - Accuracy: ~0.85 (custom model; see `metrics/metrics_custom_val.txt` for details).
  - Confusion Matrix and ROC: Available in `metrics/custom/`.
- **Limitations**: Performs best on short urban clips; may struggle with overlapping sounds or non-urban noise.
- **Ethical Considerations**: Designed for urban monitoring; ensure ethical use in surveillance contexts.
- **Saved Models**:
  - Original: `model/UrbanSound8K.keras`
  - Custom: `custom_model/custom_UrbanSound8K.keras` (and best variant)

For more details, refer to the training script and metrics outputs.

## License
MIT License.
