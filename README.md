# ECG Arrhythmia Classification

This repository contains the scaffolding for training a 1D Convolutional Neural Network (CNN) to classify ECG heartbeats (e.g., Normal vs. Arrhythmic) using TensorFlow.

## Project Structure

```
├── Dockerfile              # Docker configuration for containerized execution
├── README.md               # Project documentation
├── data/                   # Data storage
│   ├── processed/          # Processed data ready for training
│   └── raw/                # Original raw data
├── models/                 # Saved models and checkpoints
├── requirements.txt        # Python dependencies
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   │   ├── loader.py       # Data loader
│   │   └── make_dataset.py # Script to process raw data (or generate dummy data)
│   ├── models/             # Model architectures
│   │   └── cnn.py          # 1D CNN model definition
│   └── train.py            # Main training script
└── tests/                  # Unit tests
```

## Technical Approach

### 1. Model Architecture: 1D CNN
We utilize a **1D Convolutional Neural Network (CNN)**. Unlike traditional 2D CNNs used for images, 1D CNNs are highly effective for extracting features from sequential time-series data like ECGs. They can learn morphological patterns (QRS complexes, P-waves, T-waves) directly from the raw signal without manual feature engineering.

### 2. On-Device Inference: TFLite & Quantization
The model is exported to **TensorFlow Lite (TFLite)** for efficient execution on Android devices. We apply **Dynamic Range Quantization**, which converts weights to 8-bit integers while keeping activations in floating-point. This reduces model size by ~4x (from ~255KB to ~65KB) with negligible accuracy loss, critical for battery-constrained mobile use.

### 3. Data Constraints: Polar H10
The data pipeline is strictly tailored to the **Polar H10** heart rate sensor:
*   **Single Lead**: We isolate **Lead I** from the 12-lead PTB-XL dataset, as the Polar H10 simulates a single-lead ECG.
*   **Sample Rate**: We resample all training data to **130 Hz**, matching the fixed output rate of the Polar H10. This ensures the model sees the exact same signal resolution during inference as it did during training.

## Getting Started

For comprehensive instructions on environment setup, data preparation, training, and testing, please refer to the [Development Guide](DEVELOPMENT.md).

### Quick Links

- [Environment Setup](DEVELOPMENT.md#1-environment-setup)
- [Dataset Preparation](DEVELOPMENT.md#2-dataset-preparation-ptb-xl)
- [Training & Export](DEVELOPMENT.md#3-training--tflite-export)
- [Running Inference](DEVELOPMENT.md#5-running-inference)

## Roadmap

- [x] Integrate real PTB-XL dataset ingestion.
- [ ] Implement advanced preprocessing (filtering, beat segmentation).
- [ ] Tune hyperparameters for the 1D CNN.
