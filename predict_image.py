"""Runs the trained CIFAR-10 model on a single image from the command line."""

import sys
import torch
from PIL import Image
from torchvision import transforms

from cifar10_model import Cifar10CNN, CIFAR10_CLASSES, predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transform any input image into a CIFAR-10-sized normalized tensor
img_tfm = transforms.Compose([
    transforms.Resize(size=40),
    transforms.CenterCrop(size=32),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

def main(image_path):
    """
    Command-line prediction:
    python predict_image.py path/to/image.png
    """
    # Create model and load trained weights
    model = Cifar10CNN().to(DEVICE)
    model.load_state_dict(torch.load("artifacts/cifar10_cnn_best.pt", map_location=DEVICE))

    # Load image file and force to RGB
    img = Image.open(image_path).convert("RGB")

    # Transform image -> tensor and add batch dimension: (3,32,32) -> (1,3,32,32)
    x = img_tfm(img).unsqueeze(0).to(DEVICE)

    # Predict probabilities and choose top class
    probs = predict(model= model, x=x).squeeze(0)
    conf, idx = torch.max(input= probs, dim=0)

    print(f"Prediction: {CIFAR10_CLASSES[int(idx)]} | confidence={conf:.3f}")

if __name__ == "__main__":
    
    #Exactly one argument- image path
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py <image_path>")
        sys.exit(1)

    main(image_path=sys.argv[1])
