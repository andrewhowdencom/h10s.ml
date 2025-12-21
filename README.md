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

## Getting Started

### 1. Environment Setup

It is recommended to use a virtual environment or Docker.

**Using venv:**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
# Note: On Linux, you may need python3-dev for pyedflib:
# sudo apt-get install python3-dev
pip install -r requirements.txt
```

**Using Docker:**

```bash
# Build the image
docker build -t ecg-classifier .
```

### 2. Prepare Data (PTB-XL to EDF+)

The project now supports ingesting the **PTB-XL** dataset and transforming it into **EDF+** files.

1. Download and extract the PTB-XL dataset:

```bash
# Download dataset (approx 3GB)
wget -O ptb-xl-1.0.3.zip https://physionet.org/content/ptb-xl/get-zip/1.0.3/

# Unzip
unzip ptb-xl-1.0.3.zip
```

2. Run the processing script to generate `train.edf`, `val.edf`, and `test.edf`.

```bash
# Convert PTB-XL to EDF+
# Defaults to 130Hz and Lead I (Polar H10 compliant)
python src/data/make_dataset.py --input_dir ptb-xl-1.0.3 --output_dir data/processed
```

This will:
1. Extract **Lead I** (analogous to chest strap).
2. Resample data to **130 Hz** (Polar H10 sampling rate).
3. Merge recordings into continuous EDF+ files.

To train the model using the processed data:

```bash
python src/train.py --epochs 5
```

**Using Docker to train:**

```bash
docker run -v $(pwd)/models:/app/models ecg-classifier
```

### 4. Running Tests

To ensure everything is working correctly:

```bash
pytest tests/
```

## Roadmap

- [x] Integrate real PTB-XL dataset ingestion.
- [ ] Implement advanced preprocessing (filtering, beat segmentation).
- [ ] Tune hyperparameters for the 1D CNN.
