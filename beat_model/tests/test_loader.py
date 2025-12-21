import os
import shutil
import numpy as np
import sys
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.make_dataset import write_edf
from src.data.loader import load_data
from src.data.scp_codes import SCP_CODES

def test_loader_multilabel():
    # Setup
    test_dir = "data/test_verify_multilabel"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    print("Generating dummy data...")
    fs = 130 # Use 130 as per H10 spec
    duration = 10 # seconds
    n_samples = fs * duration

    # We create 3 records
    n_records = 3

    # 1 channel (Lead I)
    n_channels = 1
    total_samples = n_records * n_samples

    all_signals = []
    for i in range(n_channels):
        sig = np.random.randn(total_samples).astype(np.float32)
        all_signals.append(sig)

    # Annotations
    annotations = []
    start_time = 0.0

    # Define scenarios
    # 1. Single label: NORM
    # 2. Multi label: IMI, AMI
    # 3. No label (should be skipped or handled) -> Loader skips if no valid label found

    # Scenario 1
    desc1 = "ID:1 L:{'NORM': 100}"
    annotations.append((start_time, float(duration), desc1))
    start_time += duration

    # Scenario 2
    desc2 = "ID:2 L:{'IMI': 50, 'AMI': 50}"
    annotations.append((start_time, float(duration), desc2))
    start_time += duration

    # Scenario 3: Label not in SCP_CODES (e.g. 'XYZ')
    desc3 = "ID:3 L:{'XYZ': 100}"
    annotations.append((start_time, float(duration), desc3))
    start_time += duration

    # Write train.edf
    out_path = os.path.join(test_dir, "train.edf")
    write_edf(out_path, all_signals, fs, annotations)

    # Dummy test.edf
    shutil.copy(out_path, os.path.join(test_dir, "test.edf"))

    print("Data written. Now loading...")

    (X_train, y_train), (X_test, y_test) = load_data(test_dir)

    print(f"Loaded shapes: X_train={X_train.shape}, y_train={y_train.shape}")

    # Expected behavior:
    # Record 1: Valid.
    # Record 2: Valid.
    # Record 3: Invalid code. 'XYZ' is not in SCP_CODES. The loader should skip it (based on my implementation).
    # So we expect 2 records.

    expected_count = 2
    assert len(X_train) == expected_count
    assert X_train.shape == (expected_count, n_samples, n_channels)
    assert y_train.shape == (expected_count, 71)

    # Verify contents of y_train

    # Record 1: NORM
    norm_idx = SCP_CODES.index('NORM')
    expected_y1 = np.zeros(71)
    expected_y1[norm_idx] = 1
    np.testing.assert_array_equal(y_train[0], expected_y1)

    # Record 2: IMI, AMI
    imi_idx = SCP_CODES.index('IMI')
    ami_idx = SCP_CODES.index('AMI')
    expected_y2 = np.zeros(71)
    expected_y2[imi_idx] = 1
    expected_y2[ami_idx] = 1
    np.testing.assert_array_equal(y_train[1], expected_y2)

    print("Verification Passed!")

    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_loader_multilabel()
