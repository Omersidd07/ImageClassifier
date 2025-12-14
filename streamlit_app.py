import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from cifar10_model import Cifar10CNN, CIFAR10_CLASSES, predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")

st.title("🧠 CIFAR-10 Image Classifier")
st.write(
    "Upload an image and the model will predict one of the following CIFAR-10 object classes: "
    "Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, or Truck."
)

@st.cache_resource
def load_model():
    model = Cifar10CNN().to(DEVICE)
    model.load_state_dict(torch.load("artifacts/cifar10_cnn_best.pt", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# Transform uploaded image into the same format the model was trained on
img_tfm = transforms.Compose([
    transforms.Resize(size=40),
    transforms.CenterCrop(size=32),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

uploaded_file = st.file_uploader(label="Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=500)

    # Convert image -> tensor batch (shape: 32x32)
    x = img_tfm(image).unsqueeze(0).to(DEVICE)

    # Get probabilities for each class
    probs = predict(model=model, x=x).squeeze(0)

    # Get the most likely class & confidence
    conf, idx = torch.max(probs, dim=0)

    st.markdown("### Prediction")
    st.write(f"**Class:** {CIFAR10_CLASSES[int(idx)]}")
    st.write(f"**Confidence:** {conf.item():.2%}")

    st.markdown("### Class Probabilities")
    prob_dict = {CIFAR10_CLASSES[i]: float(probs[i]) for i in range(len(CIFAR10_CLASSES))}
    st.bar_chart(prob_dict)
