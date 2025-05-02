# AI Image Generator

This application uses OpenAI's DALL-E and Vision models to generate images based on text prompts and/or reference images. It consists of three main components:

1. **Image Generator Module**: Core functionality for image generation
2. **FastAPI Backend**: API endpoints for image generation
3. **Streamlit Frontend**: User-friendly web interface

## Features

- Generate images from text prompts
- Transform existing images using text instructions
- Support for both local image uploads and image URLs
- Adjustable parameters (model, size, quality, number of images)

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure your `.env` file contains your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Running the API

1. Start the FastAPI server:
   ```bash
   python image_api.py
   ```
   The API will be available at http://localhost:8000

2. API endpoints:
   - `/generate-from-prompt`: Generate images from text prompt only
   - `/generate-from-image`: Generate images from uploaded image and text prompt
   - `/generate-from-image-url`: Generate images from image URL and text prompt

### Running the Streamlit App

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
   The app will be available at http://localhost:8501

2. Use the web interface to:
   - Enter text prompts
   - Upload images
   - Provide image URLs
   - Adjust generation parameters
   - View and download generated images

## Examples

### Using the Image Generator Module Directly

```python
from image_generator import ImageGenerator

# Initialize generator
generator = ImageGenerator()

# Generate from text prompt only
urls = generator.generate_image("A futuristic city with flying cars and neon lights")
print(f"Generated image URLs: {urls}")

# Generate from text prompt and reference image
urls = generator.generate_image(
    prompt="Make this look like a watercolor painting", 
    image_path="path/to/image.jpg"
)
print(f"Generated image URLs: {urls}")
```

## Notes

- The application requires an active internet connection to communicate with OpenAI's API
- Image generation may take some time depending on OpenAI's server load
- DALL-E 3 provides higher quality images but may be slower than DALL-E 2
- The Vision model is used to analyze reference images and enhance prompts
