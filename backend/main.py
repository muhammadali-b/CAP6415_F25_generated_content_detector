"""
main.py

FastAPI application that exposes the AI Content Detector as a simple HTTP API.

This module:
    - Configures CORS so the Next.js frontend (running on http://localhost:3000)
      can call the backend safely.
    - Defines the `DetectionResult` response model returned to the client.
    - Provides health-check endpoints ("/" and "/ping") for debugging.
    - Implements the "/detect" endpoint, which:
        * accepts an uploaded image file,
        * validates the content type,
        * decodes it into a PIL Image,
        * forwards it to the CLIP + Logistic Regression pipeline
          (via `model_loader.predict_image`),
        * and returns the predicted label ("real" or "ai") with a confidence score.

Dependencies:
    - fastapi, uvicorn
    - pydantic
    - pillow (PIL)
    - model_loader.py (for the `predict_image` function)

Author:
    Muhammad Ali
"""

from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from model_loader import predict_image
app = FastAPI()

# Allow frontend (localhost:3000) to call this API
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionResult(BaseModel):
    """
    Response model for image detection results.

    Attributes:
        label (str): Predicted class label ("real" or "ai").
        confidence (float): Predicted probability for the chosen label
                            in the range [0.0, 1.0].
    """
    label: str
    confidence: float

# ---- Simple health endpoints ----

@app.get("/ping")
def ping():
    """
    Health-check endpoint to verify that the API is running.

    Returns:
        dict: A simple JSON message.
    """
    return {"message": "Backend is alive"}


@app.get("/")
def root():
    """
    Root endpoint providing a brief description of the API.

    Returns:
        dict: A JSON object with a short description string.
    """
    return {"message": "AI Content Detector API"}

@app.post("/detect", response_model=DetectionResult)
async def detect_image(file: UploadFile = File(...)):
    """
    Detect whether an uploaded image is AI-generated or real.

    The endpoint performs the following steps:
        1. Validate the uploaded file type.
        2. Read the file contents into memory.
        3. Open the image with PIL and convert to RGB.
        4. Use the CLIP + Logistic Regression pipeline to predict the label.
        5. Return the predicted label and confidence as JSON.

    Args:
        file (UploadFile): The uploaded image file.

    Raises:
        HTTPException: If the file type is not supported or cannot be read.

    Returns:
        DetectionResult: Predicted label, which is either "real" or "ai" and confidence.
    """
    # Basic content-type check
    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/webp",
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Please upload a JPEG, PNG, BMP, or WEBP image.",
        )

    # Reading file into memory
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded file: {e}",
        )

    # Convert raw file bytes into a PIL Image (Pillow library) so CLIP can process it.
    # PIL (Python Imaging Library) lets us open, decode, and manipulate images in Python.
    try:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse image file: {e}",
        )

    # Run prediction using the CLIP + classifier pipeline
    try:
        label, confidence = predict_image(image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {e}",
        )

    return DetectionResult(label=label, confidence=confidence)
