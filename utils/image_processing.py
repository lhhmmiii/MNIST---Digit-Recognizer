from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps


# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_SIZE = (28, 28)   # MNIST image dimensions
DIGIT_CLASSES = list(range(10))


# ─── Public API ───────────────────────────────────────────────────────────────

def preprocess_canvas(image_data: np.ndarray) -> np.ndarray:
    """Convert raw canvas data (from streamlit-drawable-canvas) to a model-ready array.

    The canvas delivers an RGBA NumPy array (height x width x 4) where the
    digit is drawn in *white* ink on a *black* background — matching the MNIST
    convention (light digit on dark background).

    Parameters
    ----------
    image_data : np.ndarray
        The ``image_data`` attribute of a ``st_canvas`` result object.

    Returns
    -------
    np.ndarray
        Shape (1, 28, 28, 1), dtype float32, values in [0, 1].

    Raises
    ------
    ValueError
        If the canvas appears to be blank (no visible strokes).
    """
    rgba = image_data.astype("uint8")

    # Compute luminance from the RGB channels (works regardless of alpha mode).
    # This correctly captures white strokes on the black canvas background.
    r, g, b = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2]
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype("uint8")

    if gray.max() == 0:
        raise ValueError("Canvas is blank — please draw a digit first.")

    # Resize to 28×28 (stroke is already white-on-black = MNIST format, no inversion needed)
    pil_img = Image.fromarray(gray, mode="L")
    pil_img = pil_img.resize(TARGET_SIZE, Image.LANCZOS)

    return _to_model_input(np.array(pil_img))


def preprocess_upload(file_bytes: bytes) -> np.ndarray:
    """Convert an uploaded image file to a model-ready array.

    Accepts any image format supported by Pillow (PNG, JPEG, BMP, etc.).
    The image is converted to grayscale, resized, and normalised.

    If the image has a *light* background with a *dark* digit (like a photo of
    handwriting on paper), pass ``invert=True`` — but by default we assume the
    image is already in MNIST style (light digit, dark background).  The
    function auto-detects this based on the mean pixel value.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded image.

    Returns
    -------
    np.ndarray
        Shape (1, 28, 28, 1), dtype float32, values in [0, 1].
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")  # grayscale
    pil_img = pil_img.resize(TARGET_SIZE, Image.LANCZOS)
    arr = np.array(pil_img)

    # Auto-invert: if image is predominantly light (mean > 127) it's likely a
    # dark-digit-on-white-background scan — invert to match MNIST convention.
    if arr.mean() > 127:
        arr = 255 - arr

    return _to_model_input(arr)


def array_to_display_image(arr_28x28: np.ndarray) -> Image.Image:
    """Scale a 28 x 28 float array back to a displayable PIL image.

    Parameters
    ----------
    arr_28x28 : np.ndarray
        2-D array of shape (28, 28) with float values in [0, 1].

    Returns
    -------
    PIL.Image.Image
        Grayscale image, upscaled to 112 x 112 for better visibility.
    """
    pixel_data = (arr_28x28 * 255).astype("uint8")
    img = Image.fromarray(pixel_data, mode="L")
    return img.resize((112, 112), Image.NEAREST)


# ─── Private helpers ──────────────────────────────────────────────────────────

def _to_model_input(arr: np.ndarray) -> np.ndarray:
    """Normalise and reshape a 28 x 28 uint8 array for model inference.

    Parameters
    ----------
    arr : np.ndarray
        Shape (28, 28), dtype uint8, values 0-255.

    Returns
    -------
    np.ndarray
        Shape (1, 28, 28, 1), dtype float32, values in [0, 1].
    """
    arr = arr.astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)
