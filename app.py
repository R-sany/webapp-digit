import streamlit as st
from generate_digits import generate_digit_images
from PIL import Image

st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("🧠 Handwritten Digit Generator")

digit = st.selectbox("Select a digit (0–9):", list(range(10)))

if st.button("Generate"):
    with st.spinner("Generating images..."):
        image_paths = generate_digit_images(digit)
        st.success(f"Generated {len(image_paths)} images of digit {digit}")
        cols = st.columns(5)
        for col, path in zip(cols, image_paths):
            col.image(Image.open(path), caption=f"Digit {digit}", use_column_width=True)
