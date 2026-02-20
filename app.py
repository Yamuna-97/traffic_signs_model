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

class_names = [
    "Speed limit (5km/h)",
    "Speed limit (15km/h)",
    "Dont Go straight",
    "Dont Go Left",
    "Dont Go Left or Right",
    "Dont Go Right",
    "Dont overtake from Left",
    "No Uturn",
    "No Car",
    "No horn",
    "No entry",
    "No stopping",
    "Speed limit (30km/h)",
    "Go straight or right",
    "Go straight",
    "Go Left",
    "Go Left or right",
    "Go Right",
    "keep Left",
    "keep Right",
    "Roundabout mandatory",
    "watch out for cars",
    "Horn",
    "Speed limit (40km/h)",
    "Bicycles crossing",
    "Uturn",
    "Road Divider",
    "Unknown6",
    "Danger Ahead",
    "Zebra Crossing",
    "Bicycles crossing",
    "Children crossing",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Speed limit (50km/h)",
    "Unknown1",
    "Unknown2",
    "Unknown3",
    "Go right or straight",
    "Go left or straight",
    "Unknown4",
    "ZigZag Curve",
    "Train Crossing",
    "Under Construction",
    "Unknown5",
    "Speed limit (60km/h)",
    "Fences",
    "Heavy Vehicle Accidents",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "Dont Go straight or left",
    "Unknown7"
]

st.title("🚦 Traffic Sign Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence_score:.2f}%")
