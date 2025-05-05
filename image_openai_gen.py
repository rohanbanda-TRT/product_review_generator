from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_TPN"))

response = client.images.edit(
  model="gpt-image-1",
  image=open("seventh.png", "rb"),
  prompt="Black and red ergonomic gaming chair in front of a triple monitor setup, RGB-lit keyboard, ambient room lighting, ideal for gamers. Person seated from behind with hands on keyboard, face not shown.",
  n=1,
  size="1024x1024",
  quality="high",
  style="natural"
)

# Save the image from base64 data
if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
    image_data = response.data[0].b64_json
    image_bytes = base64.b64decode(image_data)
    output_path = "generated_image.png"
    with open(output_path, "wb") as f:
        f.write(image_bytes)
    print(f"Image saved to {output_path}")
elif hasattr(response.data[0], 'url') and response.data[0].url:
    print(f"Image URL: {response.data[0].url}")
else:
    print("No image data found in the response")
