import os
import shutil
import numpy as np
import sys
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.make_dataset import write_edf
from src.data.loader import load_data

def test_edf_pipeline():
    # Setup
    test_dir = "data/test_verify"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    print("Generating dummy data...")
    fs = 100
    duration = 10 # seconds
    n_records = 5
    n_samples = fs * duration
    n_channels = 12
    
    all_signals = []
    # Create random signals for 12 channels, concatenated
    # We want 5 records.
    # Total samples = 5 * 1000 = 5000
    total_samples = n_records * n_samples
    
    for i in range(n_channels):
        sig = np.random.randn(total_samples).astype(np.float32)
        all_signals.append(sig)
        
    # Annotations
    annotations = []
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    start_time = 0.0
    
    expected_labels = []
    
    for i in range(n_records):
        label_str = classes[i % 5]
        expected_labels.append(i % 5)
        
        desc = f"ID:{i} L:{{'{label_str}': 100}}"
        annotations.append((start_time, float(duration), desc))
        start_time += duration
        
    # Write train.edf
    out_path = os.path.join(test_dir, "train.edf")
    write_edf(out_path, all_signals, fs, annotations)
    
    # Write test.edf (same content)
    shutil.copy(out_path, os.path.join(test_dir, "test.edf"))
    
    print("Data written. Now loading...")
    
    (X_train, y_train), (X_test, y_test) = load_data(test_dir)
    
    print(f"Loaded shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Assertions
    assert X_train.shape == (n_records, n_samples, n_channels)
    assert y_train.shape == (n_records,)
    
    # Check labels
    np.testing.assert_array_equal(y_train, np.array(expected_labels))
    
    print("Verification Passed!")
    
    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_edf_pipeline()
