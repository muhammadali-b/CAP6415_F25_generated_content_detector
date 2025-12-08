# Generated Content Detector
## CAP6415 – Computer Vision Project
### Muhammad Ali

## Abstract
The main problem addressed in this project is the increasing difficulty of distinguishing AI-generated images from real photographic content due to advances in modern generative models such as Stable Diffusion, Midjourney, and DALL·E. The objective is to develop a system capable of detecting whether a given image is AI-generated or real, using concepts covered in class like feature extraction, representation learning, linear classifiers, and convolutional architecture principles.
The problem is solved by using CLIP ViT-B/32 as a pretrained feature extractor and training a Logistic Regression classifier on top of its embeddings. A total of 100,346 images were used for the final model:

      50,026 AI-generated images, including 50000 from CIFAKE dataset and rest are self-generated and searched AI images

      50,320 real images, including CIFAKE real samples + additional real photos added manually  
        
A lightweight FastAPI backend performs inference, and a Next.js frontend allows users to upload images and view predictions. This modular system demonstrates that classical ML techniques combined with strong pretrained encoders can achieve high detection accuracy on AI-generated content.

## 1. Introduction
Advances in generative image models have made synthetic images increasingly photorealistic. As a result, distinguishing human-created content from AI-generated content has become an important challenge in computer vision.
This project builds a complete detection system that:
extracts semantic embeddings using a pretrained Vision Transformer (CLIP),


trains a binary classifier to separate real vs. synthetic images,


exposes the classifier through an accessible web interface for real-time inference.


The system follows the principles taught in the course: linear systems, feature extractors, neural representations, CNNs, Transformers, and statistical modeling.

## 2. Method Overview
### 2.1 CLIP Feature Extraction
We use the OpenAI CLIP ViT-B/32 model to convert each image into a 512-dimensional vector. CLIP acts as a pretrained feature extractor, so we do not need to train a deep model ourselves.

The feature extraction process is:

  1. Preprocess the image (resize + normalize).

  2. Pass it through CLIP’s vision encoder to obtain a feature vector.

  3. Normalize the vector so it has consistent scale.

  4. Save the embedding for training the classifier.

### 2.2 Logistic Regression Classifier
A standard Logistic Regression classifier (scikit-learn) is trained on all embedding vectors. Although simple, this classifier performs well on high-quality pretrained features.
Dataset summary:

**Class            Count**

**AI-generated**     50,026

**Real images**      50,320

**Total**          100,346


### 2.3 Backend Inference (FastAPI)
At runtime:

1- User uploads an image

2- Backend decodes the image (PIL)

3- CLIP generates a 512-dimensional vector

4- Logistic Regression outputs class probabilities

5- Backend returns the following json format:

{
  "label": "ai" | "real",
  "confidence": 0.0–1.0
}

### 2.4 Web Frontend (Next.js)

A user-friendly interface enables:

* drag-and-drop or file upload

* preview of the image

* prediction with confidence score

* clear/reset functionality

* responsive design using TailwindCSS


## 3. System Architecture
      
<img width="473" height="723" alt="Screenshot 2025-12-07 at 10 46 10 PM" src="https://github.com/user-attachments/assets/934e12cc-7364-4d8e-8808-7b41d9880f9a" />

Modules:

- ml/extract_features.py

- ml/train_classifier.py

- model_loader.py

- main.py (API)

- web/app/page.tsx (frontend)


## 4. Dataset
### 4.1 CIFAKE Dataset (Base)

CIFAKE combines:

* 50k real CIFAR-10 images,

* 50k AI-generated synthetic images (created via Stable Diffusion).

This provides a balanced and easy-to-process starting point.

## 4.2 Extended Real Dataset

To improve generalization to natural, real-world photographs, an additional 320+ personal real images were added.

Final dataset distribution:

- **Real:** 50,320

- **AI-generated:** 50,026

- **Total:** 100,346



## 5. Results
After retraining with the expanded dataset, the model achieved:

### Final Test Accuracy: 0.9354

**Classification Report**

| Class |  Precision  |  Recall  |  F1-score  |  Support  |

| ----- | ----------- | -------- | ---------- | --------- |

| real  |   0.94      |   0.93   |   0.94     |   7548    |

| ai    |   0.93      |  0.94    |   0.94     |   7504    |

**Overall metrics:**

* Accuracy: 0.94

* Macro avg: Precision 0.94, Recall 0.94, F1-score 0.94

* Weighted avg: Precision 0.94, Recall 0.94, F1-score 0.94


## 6. Installation & Execution

**Backend (FastAPI)**
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

**Feature Extraction & Training**
python ml/extract_features.py
python ml/train_classifier.py

**Frontend (Next.js)**
cd web
npm install
npm run dev

**Visit:**
http://localhost:3000


## 7. Dependencies

**Backend**

Python 3.9+
PyTorch
NumPy
Pillow
FastAPI + Uvicorn
scikit-learn
python-multipart
OpenAI CLIP

**Frontend**
Node.js 18+
React / Next.js
TailwindCSS



## 8. Attribution

This project uses:

* CLIP ViT-B/32
 Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” OpenAI, 2021.

* CIFAKE dataset
 Synthetic CIFAR-10 images generated via Stable Diffusion.

* Scikit-learn Logistic Regression
 Used for training the binary classification model.

* FastAPI documentation
 Used for structuring the backend API.

## 9. Future Work

* Continue expanding the training dataset

* Deploy the full system on a cloud server

* Extend detection to video frames

* Add mobile app support (Android/iOS)

