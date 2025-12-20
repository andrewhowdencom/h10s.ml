import tensorflow as tf
import numpy as np
from src.models.cnn import build_cnn_model

def test_model_structure():
    """
    Tests that the model builds with the correct input and output shapes.
    """
    input_shape = (1000, 1)
    num_classes = 5

    model = build_cnn_model(input_shape, num_classes)

    # Check if model is a Model instance
    assert isinstance(model, tf.keras.Model)

    # Check input shape (the first dimension is batch size, which is None)
    assert model.input_shape == (None, 1000, 1)

    # Check output shape
    assert model.output_shape == (None, num_classes)

def test_model_overfit_dummy_batch():
    """
    Tests that the model can learn from a tiny batch (sanity check).
    """
    input_shape = (100, 1)
    num_classes = 2

    model = build_cnn_model(input_shape, num_classes)

    # Create a tiny dummy batch
    X = np.random.randn(10, 100, 1)
    y = np.random.randint(0, num_classes, size=(10,))

    # Train for a few epochs
    history = model.fit(X, y, epochs=2, verbose=0)

    # Just check that it runs without error and loss exists
    assert 'loss' in history.history
