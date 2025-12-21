import argparse
import numpy as np
import tensorflow as tf
import pyedflib
import os
from src.data.scp_codes import SCP_CODES, SCP_DESCRIPTIONS

def load_tflite_model(model_path):
    """Loads TFLite model and allocates tensors."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, signal, input_shape):
    """
    Runs inference on the provided signal.
    
    Args:
        interpreter: TFLite interpreter.
        signal: 1D numpy array of ECG signal.
        input_shape: Expected input shape for the model (e.g., [1, 1000, 1]).
    
    Returns:
        List of predictions.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check input index
    input_index = input_details[0]['index']
    
    # Window size from model input shape (assuming (1, 1000, 1))
    window_size = input_shape[1]
    
    predictions = []
    
    # Simple sliding window (non-overlapping for chunks)
    # Truncate end if not full window
    num_windows = len(signal) // window_size
    
    print(f"Signal length: {len(signal)}")
    print(f"Processing {num_windows} windows of size {window_size}...")
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        chunk = signal[start:end]
        
        # Reshape to (1, 1000, 1)
        input_data = chunk.reshape(1, window_size, 1).astype(np.float32)
        
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])
        
    return np.array(predictions)

def main():
    parser = argparse.ArgumentParser(description="Run TFLite inference on EDF file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .tflite model.")
    parser.add_argument("--edf_file", type=str, required=True, help="Path to .edf file.")
    parser.add_argument("--channel_index", type=int, default=0, help="Index of channel to process (default 0).")
    
    args = parser.parse_args()
    
    # 1. Load Model
    print(f"Loading model from {args.model_path}...")
    interpreter = load_tflite_model(args.model_path)
    input_shape = interpreter.get_input_details()[0]['shape'] # e.g. [1, 1000, 1]
    print(f"Model input shape: {input_shape}")

    # 2. Load Data
    print(f"Loading data from {args.edf_file}...")
    try:
        f = pyedflib.EdfReader(args.edf_file)
        # Read signal
        signal = f.readSignal(args.channel_index)
        fs = f.getSampleFrequency(args.channel_index)
        f.close()
        print(f"Loaded signal with fs={fs}Hz")
    except Exception as e:
        print(f"Error reading EDF: {e}")
        return

    # 3. Predict
    preds = predict(interpreter, signal, input_shape)
    
    # 4. Show Results (Summary)
    # Assuming classification, we can show argmax
    if len(preds) > 0:
        classes = np.argmax(preds, axis=1)
        
        print("\nPredictions (first 20 windows):")
        for i, c in enumerate(classes[:20]):
            # Calculate time code
            start_seconds = (i * input_shape[1]) / fs
            minutes = int(start_seconds // 60)
            seconds = int(start_seconds % 60)
            time_str = f"{minutes:02}:{seconds:02}"

            code = SCP_CODES[c]
            desc = SCP_DESCRIPTIONS.get(code, "Unknown")
            print(f"Window {i} [{time_str}]: {code} ({desc})")

        unique, counts = np.unique(classes, return_counts=True)
        print("\nClass distribution:")
        for u, c in zip(unique, counts):
            code = SCP_CODES[u]
            desc = SCP_DESCRIPTIONS.get(code, "Unknown")
            print(f"Class {u} [{code}]: {c} windows - {desc}")
    else:
        print("No predictions made (signal too short?).")

if __name__ == "__main__":
    main()
