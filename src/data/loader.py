import numpy as np
import os
import pyedflib
import re
import ast

def load_edf_file(file_path):
    """
    Loads an EDF+ file and extracts segments based on annotations using pyedflib.
    Returns X (signals) and y (labels).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    f = pyedflib.EdfReader(file_path)
    
    # Read all signals
    n_channels = f.signals_in_file
    signals = []
    for i in range(n_channels):
        signals.append(f.readSignal(i))
    
    # Shape: (channels, total_samples) -> (total_samples, channels)
    full_signal = np.array(signals).T
    
    # Read annotations
    annotations = f.readAnnotations()
    # annotations is tuple (onsets, durations, descriptions)
    if len(annotations) == 3:
         onsets, durations, descriptions = annotations
    else:
         # Handle case where one might be missing or shape mismatch?
         # typically readAnnotations returns (onsets, durations, descriptions)
         onsets, durations, descriptions = annotations[0], annotations[1], annotations[2]
         
    
    X_list = []
    y_list = []
    
    fs = f.getSampleFrequency(0)
    
    # Superclass mapping
    class_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
    
    for i in range(len(onsets)):
        onset_sec = onsets[i]
        duration_sec = durations[i]
        desc = descriptions[i]
        
        # Parse description to find labels
        label_id = -1
        for k, v in class_map.items():
            if k in desc:
                label_id = v
                break 
        
        if label_id == -1:
            continue
            
        start_sample = int(onset_sec * fs)
        end_sample = int((onset_sec + duration_sec) * fs)
        
        # Extract segment
        segment = full_signal[start_sample:end_sample, :]
        
        # Ensure consistent length 
        target_len = int(10 * fs) 
        if len(segment) != target_len:
             if abs(len(segment) - target_len) < 5:
                 if len(segment) > target_len:
                     segment = segment[:target_len]
                 else:
                     pad_len = target_len - len(segment)
                     if pad_len > 0:
                        segment = np.pad(segment, ((0, pad_len), (0, 0)), 'edge')
             else:
                 continue
        
        X_list.append(segment)
        y_list.append(label_id)
        
    f.close()
    
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

def load_data(data_dir="data/processed"):
    """
    Loads processed EDF+ data from the data directory.

    Args:
        data_dir (str): Path to the processed data directory.

    Returns:
        tuple: (X_train, y_train), (X_test, y_test)
    """
    
    train_path = os.path.join(data_dir, "train.edf")
    val_path = os.path.join(data_dir, "val.edf") 
    test_path = os.path.join(data_dir, "test.edf")
    
    # For compatibility with train.py, we might concatenate train and val, or just use train
    # Let's load train and val
    print(f"Loading {train_path}...")
    try:
        X_train, y_train = load_edf_file(train_path)
    except FileNotFoundError:
         raise FileNotFoundError(f"Data not found in {data_dir}. Please run 'python src/data/make_dataset.py --input_dir <PTB-XL>' first.")

    # Optional: Load val and merge or keep separate. 
    # train.py expects (X_train, y_train), (X_test, y_test)
    # let's map: train -> train, val -> test (or actual test)
    # The user might want real test. Let's try to load test.
    
    print(f"Loading {test_path}...")
    if os.path.exists(test_path):
        X_test, y_test = load_edf_file(test_path)
    else:
        # Fallback to val if test missing
        if os.path.exists(val_path):
             X_test, y_test = load_edf_file(val_path)
        else:
             X_test, y_test = np.array([]), np.array([])

    return (X_train, y_train), (X_test, y_test)
