"""
MNIST CNN Architecture
======================
Convolutional Neural Network for MNIST handwritten digit classification.

Architecture:
    Input  : (28, 28, 1)  — grayscale images, pixel values in [0, 1]
    Layer 1: Conv2D(64, 3x3, relu, same-padding)
    Layer 2: MaxPooling2D(2x2, stride=2)
    Layer 3: Conv2D(32, 3x3, relu, same-padding)
    Layer 4: MaxPooling2D(2x2, stride=2)
    Layer 5: Flatten
    Layer 6: Dense(10, softmax)  — one logit per digit class

Expected test accuracy: ~99 % after 5 epochs on MNIST.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


def build_model() -> tf.keras.Model:
    """Build and compile the MNIST CNN model.

    Returns
    -------
    tf.keras.Model
        Compiled Keras Sequential model ready for training or inference.
    """
    model = Sequential(
        [
            # First convolutional block
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                input_shape=(28, 28, 1),
            ),
            MaxPooling2D((2, 2), strides=2),

            # Second convolutional block
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), strides=2),

            # Classifier head
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model
