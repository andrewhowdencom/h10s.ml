import os
import argparse
import tensorflow as tf
from src.data.loader import load_data
from src.models.cnn import build_cnn_model

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
    print(f"Test data shape: {X_test.shape}")

    # 2. Build Model
    input_shape = X_train.shape[1:]
    num_classes = len(set(y_train)) # Simple inference of class count
    # Fallback if labels are not 0-indexed contiguous integers, but for dummy data it's fine.
    # ideally we should pass this in or determine from dataset metadata
    if num_classes < 2: num_classes = 5 # Default fallback

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
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    # 4. Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # 5. Save Final Model
    final_model_path = "models/final_model.keras"
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ECG classification model.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed data.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")

    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size)
