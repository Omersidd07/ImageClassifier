import torch
import torch.nn as nn
import torch.nn.functional as F

# CIFAR-10: 10 classes.
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class Cifar10CNN(nn.Module):
    """
    CNN for CIFAR-10 classification:

    - Learns image features using Conv2D layers (edges/textures/shapes).
    - BatchNorm- stabilize/normalize activations (helps accuracy).
    - LeakyReLU- the activation function (non-linearity).
    - MaxPool- shrink spatial size (32x32 -> 16x16 -> 8x8).
    - Dropout- reduce overfitting.

    Note: L2 regularization is NOT defined here; it’s applied in the optimizer using weight_decay.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolution blocks that turn image into useful feature maps.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.15),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.20),
        )

        # Classifier- fully-connected layers that map features -> class scores.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes)
        )

    # Forward pass- image -> features -> logits
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def predict(model, x):
    """
    Runs inference and returns class probabilities.

    - model(x)- gives logits (unnormalized scores).
    - softmax- converts logits into probabilities that sum to 1.0.
    """
    model.eval() # put model into evaluation mode
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    return probs
