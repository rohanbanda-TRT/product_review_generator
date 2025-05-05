import base64
import sys

# Function to save base64 string to an image file
def save_base64_image(base64_string, output_path="generated_image.png"):
    try:
        # Remove any header if present (e.g., "data:image/png;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(image_data)
            
        print(f"Image successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

# If script is run directly
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Get base64 string from command line argument
        base64_string = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "generated_image.png"
        save_base64_image(base64_string, output_path)
    else:
        print("Please provide a base64 string as an argument")
        print("Usage: python save_base64_image.py <base64_string> [output_path]")
        
        # Alternative: paste your base64 string here
        base64_string = """PASTE_YOUR_BASE64_STRING_HERE"""
        
        if base64_string and base64_string != "PASTE_YOUR_BASE64_STRING_HERE":
            save_base64_image(base64_string)
