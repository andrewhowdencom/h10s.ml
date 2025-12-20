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
pip install -r requirements.txt
```

**Using Docker:**

```bash
# Build the image
docker build -t ecg-classifier .
```

### 2. Prepare Data

Currently, the project uses a script to generate dummy data for testing the pipeline. In the future, this will ingest real ECG data (e.g., PTB-XL).

```bash
# Generate dummy data
python src/data/make_dataset.py
```

### 3. Train the Model

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

- [ ] Integrate real PTB-XL dataset ingestion.
- [ ] Implement advanced preprocessing (filtering, beat segmentation).
- [ ] Tune hyperparameters for the 1D CNN.
