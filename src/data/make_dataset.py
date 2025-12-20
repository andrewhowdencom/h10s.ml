import numpy as np
import os
import argparse

def generate_dummy_data(output_dir, num_samples=1000, signal_length=1000, num_classes=5):
    """
    Generates dummy ECG data and saves it to the output directory.

    Args:
        output_dir (str): Directory to save the data.
        num_samples (int): Number of samples to generate.
        signal_length (int): Length of each ECG signal.
        num_classes (int): Number of classes.
    """
    print(f"Generating {num_samples} dummy samples...")

    # Generate random signals (samples, length, channels)
    # Using 1 channel for ECG
    X = np.random.randn(num_samples, signal_length, 1).astype(np.float32)

    # Generate random labels
    y = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)

    # Split into train and test
    split_idx = int(num_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy ECG data.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for processed data.")
    args = parser.parse_args()

    generate_dummy_data(args.output_dir)
