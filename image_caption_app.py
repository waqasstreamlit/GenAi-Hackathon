pip install --upgrade pip
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import streamlit as st


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

def generate_conditional_caption(image, text):
    inputs = processor(image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_unconditional_caption(image):
    inputs = processor(image, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    caption_type = st.selectbox("Select Caption Type:", ["Unconditional", "Conditional"])

    if caption_type == "Conditional":
        text_input = st.text_input("Enter a starting phrase for the caption:")

        if st.button("Generate Conditional Caption"):
            caption = generate_conditional_caption(image, text_input)
            st.write(f"Conditional Caption: {caption}")

    elif caption_type == "Unconditional":
        if st.button("Generate Unconditional Caption"):
            caption = generate_unconditional_caption(image)
            st.write(f"Unconditional Caption: {caption}")
