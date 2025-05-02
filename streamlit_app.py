import streamlit as st
import requests
import os
from PIL import Image
import io

# API endpoint
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Image Generator", page_icon="🖼️", layout="wide")

st.title("🖼️ AI Image Generator")
st.subheader("Generate images using OpenAI's Vision model")

# Create tabs for different generation methods
tab1, tab2, tab3 = st.tabs(["Generate from Prompt", "Upload Image", "Image URL"])

# Function to display generated images
def display_generated_images(response):
    if response.status_code == 200:
        data = response.json()
        image_urls = data.get("urls", [])
        
        if image_urls:
            st.success("Images generated successfully!")
            
            # Display each generated image
            cols = st.columns(len(image_urls))
            for i, url in enumerate(image_urls):
                with cols[i]:
                    st.image(url, caption=f"Generated Image {i+1}", use_column_width=True)
                    st.markdown(f"[Download Image]({url})")
        else:
            st.error("No images were generated.")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

# Tab 1: Generate from prompt only
with tab1:
    st.header("Generate from Text Prompt")
    
    prompt = st.text_area("Enter your prompt", height=100, 
                         placeholder="Describe the image you want to generate...")
    
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", options=["dall-e-3", "dall-e-2"], index=0)
        quality = st.selectbox("Quality", options=["standard", "hd"], index=0)
    
    with col2:
        size = st.selectbox("Size", options=["1024x1024", "1792x1024", "1024x1792"], index=0)
        n = st.slider("Number of images", min_value=1, max_value=4, value=1)
    
    if st.button("Generate Image", key="generate_prompt"):
        if prompt:
            with st.spinner("Generating image..."): 
                response = requests.post(
                    f"{API_URL}/generate-from-prompt",
                    data={
                        "prompt": prompt,
                        "model": model,
                        "size": size,
                        "quality": quality,
                        "n": n
                    }
                )
                display_generated_images(response)
        else:
            st.warning("Please enter a prompt.")

# Tab 2: Upload image
with tab2:
    st.header("Generate from Uploaded Image")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    prompt = st.text_area("Enter your prompt", height=100, key="prompt_upload",
                         placeholder="Describe how you want to transform the image...")
    
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", options=["dall-e-3", "dall-e-2"], index=0, key="model_upload")
        quality = st.selectbox("Quality", options=["standard", "hd"], index=0, key="quality_upload")
    
    with col2:
        size = st.selectbox("Size", options=["1024x1024", "1792x1024", "1024x1792"], index=0, key="size_upload")
        n = st.slider("Number of images", min_value=1, max_value=4, value=1, key="n_upload")
    
    if st.button("Generate Image", key="generate_upload"):
        if uploaded_file and prompt:
            with st.spinner("Generating image..."):
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                files = {"image": (uploaded_file.name, uploaded_file.getvalue(), f"image/{uploaded_file.type.split('/')[1]}")}
                
                response = requests.post(
                    f"{API_URL}/generate-from-image",
                    data={
                        "prompt": prompt,
                        "model": model,
                        "size": size,
                        "quality": quality,
                        "n": n
                    },
                    files=files
                )
                display_generated_images(response)
        else:
            st.warning("Please upload an image and enter a prompt.")

# Tab 3: Image URL
with tab3:
    st.header("Generate from Image URL")
    
    image_url = st.text_input("Enter image URL")
    
    if image_url:
        try:
            # Display the image from URL
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption="Image from URL", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    
    prompt = st.text_area("Enter your prompt", height=100, key="prompt_url",
                         placeholder="Describe how you want to transform the image...")
    
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", options=["dall-e-3", "dall-e-2"], index=0, key="model_url")
        quality = st.selectbox("Quality", options=["standard", "hd"], index=0, key="quality_url")
    
    with col2:
        size = st.selectbox("Size", options=["1024x1024", "1792x1024", "1024x1792"], index=0, key="size_url")
        n = st.slider("Number of images", min_value=1, max_value=4, value=1, key="n_url")
    
    if st.button("Generate Image", key="generate_url"):
        if image_url and prompt:
            with st.spinner("Generating image..."):
                response = requests.post(
                    f"{API_URL}/generate-from-image-url",
                    data={
                        "prompt": prompt,
                        "image_url": image_url,
                        "model": model,
                        "size": size,
                        "quality": quality,
                        "n": n
                    }
                )
                display_generated_images(response)
        else:
            st.warning("Please enter an image URL and a prompt.")

# Footer
st.markdown("---")
st.markdown("Built with OpenAI's DALL-E and Vision models")
