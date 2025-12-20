import numpy as np
import os

def load_data(data_dir="data/processed"):
    """
    Loads processed data from the data directory.

    Args:
        data_dir (str): Path to the processed data directory.

    Returns:
        tuple: (X_train, y_train), (X_test, y_test)
    """
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        return (X_train, y_train), (X_test, y_test)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data not found in {data_dir}. Please run 'python src/data/make_dataset.py' first.")
