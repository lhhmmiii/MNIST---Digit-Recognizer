"""
utils/inference.py
==================
Model loading and inference utilities.

Keeps all TensorFlow-specific code out of the Streamlit UI layer so that the
model can be swapped or updated independently.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# ─── Default model location ───────────────────────────────────────────────────
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "model" / "mnist_cnn.h5"


# ─── Public API ───────────────────────────────────────────────────────────────

@tf.function(reduce_retracing=True)
def _predict_fn(model, x):
    """JIT-compiled prediction for faster repeated inference."""
    return model(x, training=False)


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """Load a saved Keras model from *model_path*.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved model file (.h5 or SavedModel directory).

    Returns
    -------
    tf.keras.Model

    Raises
    ------
    FileNotFoundError
        If *model_path* does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'.\n"
            "Please train the model first:\n"
            "    python train.py\n"
        )
    return tf.keras.models.load_model(str(model_path))


def predict(
    model: tf.keras.Model,
    x: np.ndarray,
) -> dict:
    """Run inference and return a structured result dictionary.

    Parameters
    ----------
    model : tf.keras.Model
        A loaded Keras model.
    x : np.ndarray
        Pre-processed input of shape (1, 28, 28, 1).

    Returns
    -------
    dict with keys:
        ``predicted_digit`` (int)         — argmax of softmax output
        ``confidence``      (float)       — probability of the top prediction [0–1]
        ``probabilities``   (np.ndarray)  — full 10-class probability vector
        ``processing_ms``   (float)       — wall-clock inference time in ms
    """
    t0 = time.perf_counter()
    raw = model.predict(x, verbose=0)          # shape (1, 10)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    probs = raw[0]                             # shape (10,)
    predicted_digit = int(np.argmax(probs))
    confidence = float(probs[predicted_digit])

    return {
        "predicted_digit": predicted_digit,
        "confidence":      confidence,
        "probabilities":   probs,
        "processing_ms":   elapsed_ms,
    }
