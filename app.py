import streamlit as st
import torch
from torchvision.utils import save_image
from PIL import Image
import os
from datetime import datetime

from model_def import Generator  # import your Generator architecture

device = torch.device('cpu')

@st.cache_resource
def load_model():
    model = torch.load('model/my_model.pth', map_location=device)
    model.eval()
    return model

generator = load_model()

def generate_digit_images(digit, count=5):
    os.makedirs('generated', exist_ok=True)
    images = []
    for i in range(count):
        input_label = torch.zeros(1, 10, device=device)
        input_label[0, digit] = 1
        with torch.no_grad():
            generated_image = generator(input_label).cpu()
        filename = f'generated/{digit}_{i}_{datetime.now().timestamp()}.png'
        save_image(generated_image, filename)
        images.append(filename)
    return images

st.title("🧠 Handwritten Digit Generator")

digit = st.selectbox("Select a digit (0-9):", list(range(10)))

if st.button("Generate"):
    with st.spinner("Generating images..."):
        image_paths = generate_digit_images(digit)
        st.success(f"Generated {len(image_paths)} images of digit {digit}")

        cols = st.columns(5)
        for col, path in zip(cols, image_paths):
            col.image(Image.open(path), caption=f"Digit {digit}", use_column_width=True)

