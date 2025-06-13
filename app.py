import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

class_names = ['NORMAL', 'PNEUMONIA']

@st.cache_resource
def load_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

st.title("Chest X-ray Pneumonia Diagnosis")
st.write("Upload a chest X-ray image to predict if it is NORMAL or PNEUMONIA.")

model_path = "./models/resnet_chest_xray.pth"

if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please upload or move your .pth file here.")
else:
    model = load_model(model_path, len(class_names))

    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        max_width = 250
        w_percent = (max_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image_resized = image.resize((max_width, h_size), Image.LANCZOS)
        st.image(image_resized, caption='Uploaded Image')

        input_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            pred_class = class_names[preds[0]]
            st.markdown(f"### Prediction: **{pred_class}**")
