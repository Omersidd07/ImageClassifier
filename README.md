# CIFAR-10 Image Classifier

This project trains a convolutional neural network (CNN) on the CIFAR-10 image dataset and provides both command-line and web-based (Streamlit) inference.

The model classifies an image into one of the following 10 object categories:
**airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck**.

---

## Features
- Custom CNN built with PyTorch
- Convolutional layers with BatchNorm and LeakyReLU activation
- L2 regularization via AdamW optimizer
- Data augmentation (random crop + horizontal flip)
- Trained on the full CIFAR-10 training set (50,000 images)
- Achieves ~86% test accuracy
- Automatic GPU (CUDA) support if available
- Streamlit drag-and-drop demo UI
- Command-line image prediction script

---

## Project Structure
├── cifar10_model.py
├── train_cifar10.py
├── streamlit_app.py
├── predict_image.py
├── requirements.txt
├── artifacts/
└── data/

---

## Setup

### 1. Clone the repository
git clone https://github.com/your-username/cifar10-image-classifier.git
cd cifar10-image-classifier

### 2. Create and activate a virtual environment (recommended)
Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

### 3. Install dependencies
pip install -r requirements.txt
Train the Model
Run the training script: python train_cifar10.py

This will:

Download the CIFAR-10 dataset if needed
Train the CNN for 40 epochs
Evaluate accuracy on the test set after each epoch
Save model weights to: artifacts/cifar10_cnn.pt, artifacts/cifar10_cnn_best.pt

### 4. Run the Streamlit Demo

After training completes: streamlit run streamlit_app.py

Open the local URL shown in the terminal (usually http://localhost:8501).
