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
    0: "Speed limit (5km/h)",          # '0'
    1: "Speed limit (15km/h)",         # '1'
    2: "Dont Go straight",             # '10'
    3: "Dont Go Left",                 # '11'
    4: "Dont Go Left or Right",        # '12'
    5: "Dont Go Right",                # '13'
    6: "Dont overtake from Left",      # '14'
    7: "No Uturn",                     # '15'
    8: "No Car",                       # '16'
    9: "No horn",                      # '17'
    10: "No entry",                    # '18'
    11: "No stopping",                 # '19'
    12: "Speed limit (30km/h)",        # '2'
    13: "Go straight or right",        # '20'
    14: "Go straight",                 # '21'
    15: "Go Left",                     # '22'
    16: "Go Left or right",            # '23'
    17: "Go Right",                    # '24'
    18: "keep Left",                   # '25'
    19: "keep Right",                  # '26'
    20: "Roundabout mandatory",        # '27'
    21: "watch out for cars",          # '28'
    22: "Horn",                        # '29'
    23: "Speed limit (40km/h)",        # '3'
    24: "Bicycles crossing",           # '30'
    25: "Uturn",                       # '31'
    26: "Road Divider",                # '32'
    27: "Unknown6",                    # '33'
    28: "Danger Ahead",                # '34'
    29: "Zebra Crossing",              # '35'
    30: "Bicycles crossing",           # '36'
    31: "Children crossing",           # '37'
    32: "Dangerous curve to the left", # '38'
    33: "Dangerous curve to the right",# '39'
    34: "Speed limit (50km/h)",        # '4'
    35: "Unknown1",                    # '40'
    36: "Unknown2",                    # '41'
    37: "Unknown3",                    # '42'
    38: "Go right or straight",        # '43'
    39: "Go left or straight",         # '44'
    40: "Unknown4",                    # '45'
    41: "ZigZag Curve",                # '46'
    42: "Train Crossing",              # '47'
    43: "Under Construction",          # '48'
    44: "Unknown5",                    # '49'
    45: "Speed limit (60km/h)",        # '5'
    46: "Fences",                      # '50'
    47: "Heavy Vehicle Accidents",     # '51'
    48: "Speed limit (70km/h)",        # '6'
    49: "Speed limit (80km/h)",        # '7'
    50: "Dont Go straight or left",    # '8'
    51: "Unknown7"                     # '9'
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
