import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from cifar10_model import Cifar10CNN

# Choose GPU if available, otherwise use CPU. Need to figure out later how to find GPU ID.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""Make results more repeatable (same random init, same shuffles, etc.)."""
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    seed_everything()
    
    # Training hyperparameters:

    epochs = 40 # how many full passes through the training data; Change from 10 to 40 for longer learning scheduler
    batch_size = 128 # how many images per batch update
    lr = 1e-3 # learning rate
    weight_decay = 1e-4     # L2 regularization strength (applied in optimizer)
    train_subset_size = 60000  # Use full training set

    # Training transforms (data augmentation for generalization)
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])

    # Test transforms (no augmentation; only normalize)
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])

    # Downloading and loading CIFAR-10 dataset
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfms)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms)

    indices = list(range(len(train_ds)))
    random.shuffle(indices)
    train_ds = Subset(train_ds, indices[:train_subset_size])

    # DataLoaders create batches and (optionally) load data in parallel workers.
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(dataset=test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    # Build model and move it to CPU for now, until GPU gets figured out
    model = Cifar10CNN().to(DEVICE)

    # Loss function- compares predicted logits vs true class labels.
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer- updates model weights. weight_decay = L2 regularization.
    optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # Schedule- slowly changes the learning rate across training? WILL INCREASE TRAINING TIME?
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    
    def evaluate():
        """Evaluate accuracy on the test set/data"""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total
    
    # Folder path for pytorch model (weights).
    os.makedirs("artifacts", exist_ok=True)

    """Save best model"""
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        model.train() # training mode enables dropout and BatchNorm updates
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Reset gradients from previous step
            optimizer.zero_grad()

            # Forward pass -> compute loss -> backprop -> update weights
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        acc = evaluate()
        scheduler.step()

        print(f"Epoch {epoch}/{epochs} | loss={running_loss/len(train_loader):.4f} | test_acc={acc:.4f}")

        # save best checkpoint epoch
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), "artifacts/cifar10_cnn_best.pt")

    # save final model
    torch.save(model.state_dict(), "artifacts/cifar10_cnn.pt")
    print("Model saved to artifacts/cifar10_cnn.pt")
    print(f"Best model saved to artifacts/cifar10_cnn_best.pt (epoch {best_epoch}, acc={best_acc:.4f})")

if __name__ == "__main__":
    main()
