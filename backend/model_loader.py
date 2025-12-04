from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
from PIL import Image
import clip

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
MODEL_PATH = MODELS_DIR / "clip_logreg_detector.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model():
    """
    Load the CLIP ViT-B/32 model and its preprocessing pipeline.

    The model is moved to the selected device (CPU or GPU) and set
    to evaluation mode.

    Returns:
        Tuple[torch.nn.Module, callable]: A tuple (model, preprocess) where
            - model: CLIP image encoder.
            - preprocess: preprocessing function for PIL images.
    """
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def load_classifier():
    """
    Load the trained Logistic Regression classifier from disk.

    The classifier is expected to have been trained on CLIP embeddings
    and saved to MODEL_PATH.

    Raises:
        FileNotFoundError: If the model file does not exist.

    Returns:
        Any: The loaded classifier object.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find trained model at {MODEL_PATH}. "
            f"Make sure you have run train_classifier.py."
        )
    clf = joblib.load(MODEL_PATH)
    return clf


# Load models once so they are reused across requests
clip_model, clip_preprocess = load_clip_model()
classifier = load_classifier()


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for CLIP and return a batched tensor.

    Args:
        image (PIL.Image.Image): Input image in RGB format.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, C, H, W)
                      on the correct device.
    """
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    return image_tensor


def extract_clip_embedding(image: Image.Image) -> np.ndarray:
    """
    Compute a normalized CLIP embedding for a given image.

    The embedding is:
        - Computed using CLIP's image encoder.
        - L2-normalized to match the training pipeline.

    Args:
        image (PIL.Image.Image): Input image in RGB format.

    Returns:
        np.ndarray: 1D numpy array representing the image embedding.
    """
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        features = clip_model.encode_image(image_tensor)
        # Normalize to unit length (same as in extract_features.py)
        features = features / features.norm(dim=-1, keepdim=True)

    embedding = features.cpu().numpy().squeeze()
    return embedding


def predict_image(image: Image.Image) -> Tuple[str, float]:
    """
    Predict whether an image is AI-generated or real.

    Steps:
        1. Compute CLIP embedding for the image.
        2. Run the trained classifier on the embedding.
        3. Interpret probabilities and map to a human-readable label.

    Args:
        image (PIL.Image.Image): Input image in RGB format.

    Returns:
        Tuple[str, float]: label, confidence
            - label: "real" or "ai"
            - confidence: predicted probability for the chosen label.
    """
    embedding = extract_clip_embedding(image)
    # Classifier expects shape (1, D)
    x = embedding.reshape(1, -1)

    # predict_proba gives probabilities for each class [p(real), p(ai)]
    probs = classifier.predict_proba(x)[0]
    prob_real = float(probs[0])
    prob_ai = float(probs[1])

    if prob_ai >= prob_real:
        label = "ai"
        confidence = prob_ai
    else:
        label = "real"
        confidence = prob_real

    return label, confidence
