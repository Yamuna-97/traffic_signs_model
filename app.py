import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

device = torch.device("cpu")

class Image_CNN(nn.Module):
    def __init__(self, num_classes=52):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 600),
            nn.ReLU(),
            nn.Linear(600, 120),
            nn.ReLU(),
            nn.Linear(120, 52)
        )

    def forward(self, x):
        x = self.network(x)
        x = self.fc_layer(x)
        return x


model = Image_CNN()
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

number_to_name = {
    0: "Speed limit (5km/h)",
    # paste FULL dictionary here
}

st.title("🚦 Traffic Sign Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)
        confidence, predicted = torch.max(prob, 1)

    st.success(f"Prediction: {number_to_name[predicted.item()]}")
    st.info(f"Confidence: {confidence.item()*100:.2f}%")
