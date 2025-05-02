import os
import base64
import requests
from typing import Optional, List, Union
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ImageGenerator:
    """Class to handle image generation using OpenAI's DALL-E and Vision models"""
    
    def __init__(self):
        self.client = client
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string
        
        Args:
            image_path: Path to the image file or URL
            
        Returns:
            Base64 encoded image string
        """
        # If image_path is a URL, download the image first
        if image_path.startswith('http'):
            response = requests.get(image_path)
            image_data = response.content
            image = Image.open(io.BytesIO(image_data))
        else:
            # Open local image file
            image = Image.open(image_path)
        
        # Convert to RGB if needed (in case of RGBA or other formats)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save to bytes buffer
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        
        # Encode to base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_image(self, prompt: str, image_path: Optional[str] = None, 
                      model: str = "dall-e-3", size: str = "1024x1024", 
                      quality: str = "standard", n: int = 1) -> List[str]:
        """Generate image based on prompt and optionally a reference image
        
        Args:
            prompt: Text prompt for image generation
            image_path: Optional path to reference image or image URL
            model: OpenAI model to use (dall-e-3 by default)
            size: Image size (1024x1024, 1792x1024, or 1024x1792)
            quality: Image quality (standard or hd)
            n: Number of images to generate
            
        Returns:
            List of URLs to generated images
        """
        try:
            if image_path:
                # If image is provided, use Vision model for context
                base64_image = self.encode_image(image_path)
                
                # First get context from the image using GPT-4 Vision
                vision_response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an assistant that helps generate detailed image descriptions based on reference images and user prompts."},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"Based on this image and the following prompt, create a detailed description for DALL-E to generate a new image: {prompt}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ],
                    max_tokens=300
                )
                
                # Extract the enhanced prompt
                enhanced_prompt = vision_response.choices[0].message.content
                
                # Now use DALL-E with the enhanced prompt
                response = self.client.images.generate(
                    model=model,
                    prompt=enhanced_prompt,
                    size=size,
                    quality=quality,
                    n=n
                )
            else:
                # If no image, use DALL-E directly
                response = self.client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=n
                )
            
            # Return URLs of generated images
            return [image.url for image in response.data]
        
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    generator = ImageGenerator()
    # Test with just a prompt
    urls = generator.generate_image("A futuristic city with flying cars and neon lights")
    print(f"Generated image URLs: {urls}")
    
    # Test with a prompt and reference image (if available)
    # urls = generator.generate_image("Make this look like a watercolor painting", "path/to/image.jpg")
    # print(f"Generated image URLs: {urls}")
