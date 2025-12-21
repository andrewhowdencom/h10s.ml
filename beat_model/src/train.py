import os
import argparse
import subprocess
import numpy as np
import tensorflow as tf
from src.data.loader import load_data
from src.models.cnn import build_cnn_model
from src.data.scp_codes import SCP_CODES

def train(data_dir, epochs, batch_size):
    """
    Main training function.
    """
    # 1. Load Data
    print("Loading data...")
    try:
        (X_train, y_train), (X_test, y_test) = load_data(data_dir)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Training data shape: {X_train.shape}")
    if len(X_test) > 0:
        print(f"Test data shape: {X_test.shape}")

    if len(X_train) == 0:
        print("No training data found. Exiting.")
        return

    # 2. Build Model
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    print(f"Building model with input shape {input_shape} and {num_classes} classes...")
    model = build_cnn_model(input_shape, num_classes)
    model.summary()

    # 3. Callbacks
    checkpoint_path = "models/model_checkpoint.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    # 4. Train
    print("Starting training (with Focal Loss)...")

    validation_data = (X_test, y_test) if len(X_test) > 0 else None

    # Note: We removed manual sample_weight in favor of Focal Loss in the model definition
    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # 5. Save Final Model
    final_model_path = "models/final_model.keras"
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # 6. Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Get git hash
    def get_git_version():
        try:
            cmd = "git rev-parse --short HEAD"
            revision = subprocess.check_output(cmd.split()).strip().decode('utf-8')
            cmd = "git status --porcelain"
            status = subprocess.check_output(cmd.split()).strip().decode('utf-8')
            if status:
                revision += "-dirty"
            return revision
        except Exception:
            return "unknown"

    git_version = get_git_version()
    tflite_model_path = f"models/model_{git_version}.tflite"
    
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ECG classification model.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed data.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")

    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size)
