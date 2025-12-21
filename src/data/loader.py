import numpy as np
import os
import pyedflib
import re
import ast
from src.data.scp_codes import SCP_CODES

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
    
    # Create a mapping from code to index
    code_to_index = {code: i for i, code in enumerate(SCP_CODES)}
    
    for i in range(len(onsets)):
        onset_sec = onsets[i]
        duration_sec = durations[i]
        desc = descriptions[i]
        
        # Parse description to find labels
        # desc is like "ID:123 L:{'NORM': 100.0, 'LBBB': 50.0}"
        y_vector = np.zeros(len(SCP_CODES), dtype=np.int32)
        found_valid_label = False
        
        # Regex to find the dictionary part after "L:"
        match = re.search(r"L:({.*})", desc)
        if match:
            try:
                labels_dict_str = match.group(1)
                labels_dict = ast.literal_eval(labels_dict_str)

                for code in labels_dict.keys():
                    if code in code_to_index:
                        y_vector[code_to_index[code]] = 1
                        found_valid_label = True
            except:
                pass

        if not found_valid_label:
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
        y_list.append(y_vector)
        
    f.close()
    
    if len(X_list) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

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
    
    print(f"Loading {train_path}...")
    try:
        X_train, y_train = load_edf_file(train_path)
    except FileNotFoundError:
         raise FileNotFoundError(f"Data not found in {data_dir}. Please run 'python src/data/make_dataset.py --input_dir <PTB-XL>' first.")

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
