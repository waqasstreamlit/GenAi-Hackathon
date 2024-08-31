import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

# Your Gemini API key
GEMINI_API_KEY = 'AIzaSyAE__BHdLtFBn3HEi0EcJ7ySNkZvqRU5YA'

def generate_detailed_description(image):
    inputs = processor(image, "text", return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

def enhance_description_with_gemini(description):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f"Based on the following description of an image, provide a more detailed description, "
                            f"generate possible captions, and suggest logo ideas. \n\n"
                            f"Description: {description}\n\n"
                            f"Detailed Description:\n"
                            f"Captions:\n"
                            f"Logo Suggestions:\n"
                        )
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for HTTP error responses
        response_json = response.json()
        
        # Extract the text from the response
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            candidate = response_json["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0:
                    return parts[0].get("text", "No response text found")
        
        return "No contents or parts in response"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

# Streamlit app interface
st.title("Image Caption and Enhancement Tool")

# Image uploader or URL input
upload_option = st.selectbox("Upload Image or Provide Image URL", ["Upload Image", "Provide Image URL"])

if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Generate and display the description
        initial_description = generate_detailed_description(image)
        st.subheader("Initial Description")
        st.write(initial_description)
        # Enhance and display the description
        enhanced_text = enhance_description_with_gemini(initial_description)
        st.subheader("Enhanced Description, Captions, and Logo Suggestions")
        st.write(enhanced_text)

elif upload_option == "Provide Image URL":
    img_url = st.text_input("Enter Image URL")
    if img_url:
        try:
            response = requests.get(img_url, stream=True)
            image = Image.open(response.raw).convert('RGB')
            st.image(image, caption="Image from URL", use_column_width=True)
            # Generate and display the description
            initial_description = generate_detailed_description(image)
            st.subheader("Initial Description")
            st.write(initial_description)
            # Enhance and display the description
            enhanced_text = enhance_description_with_gemini(initial_description)
            st.subheader("Enhanced Description, Captions, and Logo Suggestions")
            st.write(enhanced_text)
        except Exception as e:
            st.error(f"Error loading image: {e}")
