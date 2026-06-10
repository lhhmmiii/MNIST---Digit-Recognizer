"""
train.py — Train and save the MNIST CNN model
==============================================

Usage
-----
    python train.py                  # default 5 epochs, saves to model/mnist_cnn.h5
    python train.py --epochs 10      # custom epoch count
    python train.py --out my_model.h5

The script downloads MNIST automatically via Keras if not already cached.
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from model.architecture import build_model

# ─── Default paths ────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "mnist_cnn.h5")


def load_and_preprocess_data():
    """Load MNIST and apply the same preprocessing used in the notebook.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test) with shapes:
            X : (N, 28, 28, 1)  float32 in [0, 1]
            y : (N, 10)         one-hot encoded
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape to (N, H, W, C) and normalise to [0, 1]
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test  = to_categorical(y_test,  num_classes=10)

    return X_train, y_train, X_test, y_test


def train(epochs: int = 5, model_path: str = DEFAULT_MODEL_PATH) -> None:
    """Train the model and save it to *model_path*.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    model_path : str
        Destination path for the saved model (HDF5 / SavedModel format).
    """
    print("=" * 60)
    print("  MNIST CNN — Training Script")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1/3] Loading and preprocessing data …")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    print(f"      Train : {X_train.shape}  |  Test : {X_test.shape}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/3] Building model …")
    model = build_model()
    model.summary()

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[3/3] Training for {epochs} epoch(s) …")
    t0 = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
    )

    elapsed = time.time() - t0
    print(f"\n      Training finished in {elapsed:.1f}s")

    # ── Evaluation ────────────────────────────────────────────────────────────
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"      Test  loss     : {loss:.4f}")
    print(f"      Test  accuracy : {accuracy * 100:.2f}%")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
    print(f"\n      Model saved → {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MNIST CNN and save the model.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Output path for the saved model (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    train(epochs=args.epochs, model_path=args.out)
