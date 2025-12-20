import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape, num_classes):
    """
    Builds a 1D Convolutional Neural Network for ECG classification.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, channels).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    inputs = layers.Input(shape=input_shape)

    # First Convolutional Block
    x = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Second Convolutional Block
    x = layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Third Convolutional Block
    x = layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Global Pooling and Dense Layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ECG_CNN_Model")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
