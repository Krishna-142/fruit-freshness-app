import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# ------------------ MODEL CLASS ------------------

import requests

import requests

def download_model():
    url = "https://drive.google.com/uc?export=download&id=16Q_GSLrxlGTtKWjb67cjNkeZpdi2dV4j"
    
    response = requests.get(url, stream=True)
    
    with open("fruit_model.pth", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
import os

if not os.path.exists("fruit_model.pth") or os.path.getsize("fruit_model.pth") < 1000000:
    download_model()


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Base model (NO weight loading here)
        self.base = models.resnet18(weights=None)

        # Remove final layer
        self.base.fc = nn.Identity()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        # Block 2 (Fruit)
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5)
        )

        # Block 3 (Freshness)
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        y1 = self.block2(x)
        y2 = self.block3(x)
        return y1, y2


# ------------------ LOAD MODEL ------------------
model = Model()
model.load_state_dict(torch.load("fruit_model.pth", map_location="cpu"))
model.eval()

# ------------------ LABELS ------------------
fruit_classes = ['Apple', 'Banana', 'Orange', 'Potato', 'Tomato']

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# ------------------ SUGGESTION FUNCTION ------------------
def get_suggestion(fruit, freshness):
    if freshness < 40:
        return "❌ Not fit to eat"

    if fruit == "Potato":
        return "🥔 Store in a cool, dark place (not fridge)"
    elif fruit == "Banana":
        return "🍌 Keep at room temperature"
    elif fruit == "Apple":
        return "🍎 Store in refrigerator"
    elif fruit == "Tomato":
        return "🍅 Keep outside fridge"
    else:
        return "✔️ Store properly"


# ------------------ STREAMLIT UI ------------------
st.title("🍎 Fruit Freshness Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
image_url = st.text_input("Or paste image URL here")

import requests
from io import BytesIO
import torch.nn.functional as F
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-image: url("https://t3.ftcdn.net/jpg/03/09/90/26/360_F_309902682_6Skuz4I4IjOBqHEdbQXam6tUbV5hVZLR.jpg");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#     }

#     /* Blur + white overlay */
#     .stApp::before {
#         content: "";
#         position: fixed;
#         top: 0;
#         left: 0;
#         width: 100%;
#         height: 100%;
#         backdrop-filter: blur(10px);   /* 👈 blur strength */
#         background: rgba(255, 255, 255, 0.5);
#         z-index: -1;
#     }

#     /* Force all text black */
#     h1, h2, h3, h4, h5, h6, p, label, div {
#         color: black !important;
#     }

#     .stTextInput label, .stFileUploader label {
#         color: black !important;
#         font-weight: 600;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
image = None

# Case 1: Upload
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

# Case 2: URL
elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except:
        st.error("❌ Could not load image from URL")

# Run prediction if image is available
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    img = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        fruit_pred, fresh_pred = model(img)

    # Fruit prediction
    fruit_idx = torch.argmax(fruit_pred, dim=1).item()
    fruit = fruit_classes[fruit_idx]

    # Freshness calculation (real probabilities)
    probs = F.softmax(fresh_pred, dim=1)
    fresh_conf = probs[0][0].item()
    spoiled_conf = probs[0][1].item()

    status_idx = torch.argmax(fresh_pred, dim=1).item()

    if status_idx == 1:
        freshness = (1 - spoiled_conf) * 100
        status = "Rotten"
    else:
        freshness = fresh_conf * 100
        status = "Fresh"

    freshness = round(freshness, 2)

    suggestion = get_suggestion(fruit, freshness)

    # Output
    st.write(f"### 🍏 Fruit: {fruit}")
    st.write(f"### 🌡️ Freshness: {freshness}%")
    st.write(f"### 🟢 Status: {status}")
    st.write(f"### 💡 Suggestion: {suggestion}")
    st.progress(int(freshness))
