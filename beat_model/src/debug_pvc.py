
import os
import argparse
import numpy as np
import tensorflow as tf
import pyedflib
from src.data.scp_codes import SCP_CODES
from src.data import processing

def load_tflite_model(model_path):
    try:
        # Try importing from tflite_runtime first (for edge devices)
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
    except ImportError:
        # Fallback to full tensorflow
        interpreter = tf.lite.Interpreter(model_path=model_path)
        
    interpreter.allocate_tensors()
    return interpreter

def load_simple_edf(edf_file):
    if not os.path.exists(edf_file):
        raise FileNotFoundError(f"{edf_file} does not exist.")
    
    f = pyedflib.EdfReader(edf_file)
    # Assume 1st channel is the one we want
    signal = f.readSignal(0)
    fs = f.getSampleFrequency(0)
    f.close()
    return signal, fs

def predict(interpreter, signal, input_shape):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_index = input_details[0]['index']
    window_size = input_shape[1]
    
    predictions = []
    num_windows = len(signal) // window_size
    
    print(f"Processing windows of size {window_size} with stride {window_size//2}...")
    
    step = window_size // 2
    # Ensure we don't go out of bounds
    num_windows = (len(signal) - window_size) // step + 1
    
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        chunk = signal[start:end]
        
        input_data_flat = chunk.astype(np.float32)
        
        # Apply Filters (Important: Must match training)
        # Apply Preprocessing (Filter + Normalize)
        input_data_flat = processing.apply_filters(input_data_flat, 130.0)
        input_data_flat = processing.normalize_signal(input_data_flat)
        
        input_data = input_data_flat.reshape(1, window_size, 1).astype(np.float32)        
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])
        
    return np.array(predictions)

def main():
    parser = argparse.ArgumentParser(description='Debug PVC detection')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .tflite model')
    parser.add_argument('--edf_file', type=str, required=True, help='Path to .edf file')
    args = parser.parse_args()
    
    # 1. Load Model
    print(f"Loading model from {args.model_path}...")
    interpreter = load_tflite_model(args.model_path)
    input_shape = interpreter.get_input_details()[0]['shape']
    
    # 2. Load Data
    print(f"Loading data from {args.edf_file}...")
    try:
        signal, fs = load_simple_edf(args.edf_file)
        print(f"Loaded signal with fs={fs}Hz")
        print(f"Signal length: {len(signal)}")
        print(f"Stats: Min={np.min(signal):.4f}, Max={np.max(signal):.4f}, Mean={np.mean(signal):.4f}, Std={np.std(signal):.4f}")
        
        # Check training data stats for comparison
        train_edf = "data/processed/train.edf"
        if os.path.exists(train_edf):
            print(f"\n--- Comparing with Training Data ({train_edf}) ---")
            try:
                t_signal, t_fs = load_simple_edf(train_edf)
                # Take first 10000 samples
                t_signal = t_signal[:10000] 
                print(f"Train Stats (first 10k): Min={np.min(t_signal):.4f}, Max={np.max(t_signal):.4f}, Mean={np.mean(t_signal):.4f}, Std={np.std(t_signal):.4f}")
            except Exception as e:
                print(f"Could not load training data for comparison: {e}")
        else:
             print(f"\nTraining data {train_edf} not found. Cannot compare stats.")

    except Exception as e:
        print(f"Error loading EDF: {e}")
        return

    # 3. Predict
    preds = predict(interpreter, signal, input_shape)
    
    # 4. Analyze PVC (Index 45)
    pvc_index = 45 # Verified visually
    print(f"\n--- PVC Analysis (Class {pvc_index}: {SCP_CODES[pvc_index]}) ---")
    
    if len(preds) > 0:
        for i, p in enumerate(preds):
            # Calculate timestamp
            timestamp = i * 5 
            mm_ss = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"

            # Get top 10 classes
            top_10_indices = np.argsort(p)[-10:][::-1]
            
            print(f"Window {i} [{mm_ss}]:")
            for idx in top_10_indices:
                prob = p[idx]
                name = SCP_CODES[idx]
                star = "*" if idx == pvc_index else " "
                print(f"  {star} {name:<10}: {prob*100:.2f}%")
            print("-" * 30)
    else:
        print("No predictions made.")

if __name__ == "__main__":
    main()
