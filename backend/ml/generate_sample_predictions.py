"""
generate_sample_predictions.py

Generate a grid of sample images (real + AI) with their predicted labels
and confidence scores, and save it as results/sample_predictions.png.

Expected folder structure:

    results_samples/
        real/
        ai/

You can change the filenames; the script will automatically pick up to
3 images from each folder.
"""

from pathlib import Path
from typing import List
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure the backend directory is on sys.path so we can import model_loader
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model_loader import predict_image  # uses your existing CLIP + Logistic Regression pipeline

SAMPLES_REAL_DIR = BACKEND_DIR / "results_samples" / "real"
SAMPLES_AI_DIR = BACKEND_DIR / "results_samples" / "ai"

RESULTS_DIR = BACKEND_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "sample_predictions.png"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    """
    Return a sorted list of image files in the given folder.

    Args:
        folder (Path): Folder containing images.

    Returns:
        List[Path]: List of image paths.
    """
    if not folder.exists():
        return []
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def main():
    # Collect sample images
    real_images = list_images(SAMPLES_REAL_DIR)
    ai_images = list_images(SAMPLES_AI_DIR)

    if not real_images and not ai_images:
        raise RuntimeError(
            f"No images found in:\n  {SAMPLES_REAL_DIR}\n  {SAMPLES_AI_DIR}\n"
            "Please add a few real and AI images there first."
        )

    real_samples = real_images[:3]
    ai_samples = ai_images[:3]

    samples = real_samples + ai_samples
    true_labels = ["real"] * len(real_samples) + ["ai"] * len(ai_samples)

    num_samples = len(samples)
    if num_samples == 0:
        raise RuntimeError("No sample images were found.")

    # Grid layout: e.g., 2 rows x 3 columns for 6 images
    cols = 3
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Normalize axes to a flat array
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    axes = axes.flatten()

    for ax, img_path, true_label in zip(axes, samples, true_labels):
        img = Image.open(img_path).convert("RGB")

        pred_label, confidence = predict_image(img)

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"{img_path.name}\nTrue: {true_label} | Pred: {pred_label} ({confidence:.2f})",
            fontsize=8,
        )

    # Turn off any empty axes if fewer than rows*cols images
    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved sample predictions grid to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
