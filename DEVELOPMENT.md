# Development Guide

This guide outlines the workflow for developing, training, and exporting the ECG Arrhythmia Classification model.

## 1. Environment Setup

Ensure you have Python 3 installed.

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate

# Install dependencies (requires python3-dev for pyedflib)
pip install -r requirements.txt
```

# TL, DR

```bash
source .venv/bin/activate && \
    aws s3 sync \
    --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ ptb-xl-1.0.3 \
    && .venv/bin/python -m src.data.make_dataset \
        --input_dir ptb-xl-1.0.3 \
        --output_dir data/processed \
        --target_fs 130
    && python -m src.train --epochs 10 --batch_size 32
```

## 2. Dataset Preparation (PTB-XL)

The model expects data in **EDF+** format. We use the PTB-XL dataset as the source.

**Constraint**: The target hardware is the **Polar H10** (130 Hz, Lead I). The processing script enforces this by default.

```bash
# 1. Download PTB-XL
wget -O ptb-xl-1.0.3.zip https://physionet.org/content/ptb-xl/get-zip/1.0.3/
unzip ptb-xl-1.0.3.zip

# 2. Convert to compliant EDF+
# - Extracts Lead I
# - Resamples to 130 Hz
python -m src.data.make_dataset --input_dir ptb-xl-1.0.3 --output_dir data/processed
```

## 3. Training & TFLite Export

Run the training script to train the CNN and automatically export a TFLite model.

```bash
python -m src.train --epochs 10 --batch_size 32
```

### Outputs
*   **Keras Model**: `models/final_model.keras`
*   **TFLite Model**: `models/model_<git-hash>.tflite` (e.g., `models/model_a1b2c3d.tflite`)

**Note**: The TFLite model uses **Dynamic Range Quantization** (Int8 weights, Float32 activations) for optimized on-device performance on Android.

## 4. Running Tests

To verify the data pipeline integrity:

```bash
pytest tests/
```

## 5. Running Inference

### 5.1. CLI (Python)

You can run inference on an EDF file using the `src/predict.py` script. This is useful for validating the model on a desktop environment before deploying to mobile.

```bash
# General usage
python -m src.predict --model_path models/model_<hash>.tflite --edf_file data/processed/test.edf

# Example
python -m src.predict --model_path models/model_final.tflite --edf_file data/processed/test.edf
```

### 5.2. Android (Kotlin)

To run inference in the Android app, use the TensorFlow Lite Interpreter API.

**Dependencies (`build.gradle.kts`):**
```kotlin
implementation("org.tensorflow:tensorflow-lite:2.14.0")
implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
```

**Inference Implementation:**

```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.FileInputStream
import java.nio.channels.FileChannel

class ECGResultAnalyzer(private val context: Context) {

    private var interpreter: Interpreter? = null

    init {
        interpreter = Interpreter(loadModelFile("model.tflite"))
    }

    private fun loadModelFile(modelPath: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun predict(ecgSignal: FloatArray): FloatArray {
        // 1. Prepare Input
        // Model expects [1, 1000, 1] - Float32
        // Input size: 1000 * 4 bytes = 4000 bytes
        val inputBuffer = ByteBuffer.allocateDirect(1000 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        // Fill buffer with signal data
        for (sample in ecgSignal) {
            inputBuffer.putFloat(sample)
        }

        // 2. Prepare Output
        // Assuming 5 classes --> [1, 5]
        val outputBuffer = Array(1) { FloatArray(5) }

        // 3. Run Inference
        interpreter?.run(inputBuffer, outputBuffer)

        // Return probabilities
        return outputBuffer[0]
    }

    fun close() {
        interpreter?.close()
    }
}
```
