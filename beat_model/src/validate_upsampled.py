import argparse
import numpy as np
import pyedflib
import wfdb.processing
import tensorflow as tf
from scipy import signal as scipy_signal
from src.data import processing

# AAMI Mapping for display
AAMI_LABELS = {0: 'N (Normal)', 1: 'S (Supraventricular)', 2: 'V (Ventricular/PVC)', 3: 'F (Fusion)', 4: 'Q (Unknown)'}

def load_h10_edf(file_path):
    f = pyedflib.EdfReader(file_path)
    sig = f.readSignal(0)
    fs = f.getSampleFrequency(0)
    f.close()
    return sig, fs

def upsample_signal(sig, original_fs, target_fs=360.0):
    num_samples = int(len(sig) * target_fs / original_fs)
    resampled_sig = scipy_signal.resample(sig, num_samples)
    return resampled_sig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model {args.model_path}...")
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=args.model_path)
    except:
        interpreter = tf.lite.Interpreter(model_path=args.model_path)
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # 2. Load and Upsample Data
    print(f"Loading {args.edf_file}...")
    sig, fs = load_h10_edf(args.edf_file)
    print(f"Original: {len(sig)} samples @ {fs}Hz")
    
    # Robust Scale BEFORE Resampling? Or After?
    # Usually better to clean first.
    sig = processing.normalize_signal(sig) # Robust scale the raw 130Hz
    
    target_fs = 360.0
    print(f"Upsampling to {target_fs}Hz...")
    sig_360 = upsample_signal(sig, fs, target_fs)
    print(f"Upsampled: {len(sig_360)} samples")

    # 3. Detect Beats (QRS Detection)
    print("Detecting R-peaks (XQRS)...")
    qrs_inds = wfdb.processing.xqrs_detect(sig=sig_360, fs=target_fs, verbose=False)
    print(f"Detected {len(qrs_inds)} beats.")

    # 4. Extract and Classify Beats
    window_size = 256
    results = []
    
    print("Classifying beats...")
    for r_peak in qrs_inds:
        start = r_peak - window_size // 2
        end = r_peak + window_size // 2
        
        if start < 0 or end > len(sig_360):
            continue
            
        beat = sig_360[start:end]
        
        # Prepare for TFLite
        input_data = beat.reshape(1, window_size, 1).astype(np.float32)
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)[0]
        
        pred_class = np.argmax(output_data)
        confidence = output_data[pred_class]
        
        results.append({
            'time': r_peak / target_fs,
            'class': pred_class,
            'conf': confidence,
            'probs': output_data
        })

    # 5. Report
    print("\n--- Classification Results ---")
    counts = {}
    for r in results:
        cls = r['class']
        counts[cls] = counts.get(cls, 0) + 1
        
        # Print if Abnormal (not 0) or high PVC prob
        if cls != 0 or r['probs'][2] > 0.1: # Class 2 is PVC
            mm_ss = f"{int(r['time']//60):02d}:{int(r['time']%60):02d}"
            label = AAMI_LABELS[cls]
            pvc_prob = r['probs'][2]
            print(f"[{mm_ss}] Pred: {label:<20} | PVC Prob: {pvc_prob*100:.1f}%")

    print(f"\nSummary counts: {counts}")
    print("Mapping: 0=N, 1=S, 2=V(PVC), 3=F, 4=Q")

if __name__ == "__main__":
    main()
