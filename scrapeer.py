#!/usr/bin/env python3
"""
Amazon Product Content Generation System
----------------------------------
A script to scrape Amazon product details and generate content using OpenAI.

Usage:
    python scrapeer.py <amazon_product_url> [--output filename.json]
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import argparse
import sys
import os
import time
import random
from typing import Dict, List, Any, Optional, Union
import openai
from dotenv import load_dotenv
import base64
from PIL import Image
from io import BytesIO
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scrapeer.log")
    ]
)
logger = logging.getLogger(__name__)

# Import the more reliable scraping function from bs4scrape.py
from bs4scrape import get_amazon_product_details as bs4_get_amazon_product_details

# Load environment variables from .env file
load_dotenv()

# Check if OpenAI API key is set in environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set it in your .env file.")

openai.api_key = OPENAI_API_KEY


def fetch_amazon_product_details(url: str) -> Dict[str, Any]:
    """
    Fetches product details from an Amazon product URL
    
    Args:
        url (str): Amazon product URL
        
    Returns:
        dict: Product details including title, price, description, features, and images
    """
    # List of user agents to rotate
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    ]
    
    # Headers to mimic a browser visit
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': 'https://www.google.com/',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Send request to the Amazon URL
        print(f"Fetching data from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check if the request was successful
        if response.status_code != 200:
            return {"error": f"Failed to retrieve the page. Status code: {response.status_code}"}
        
        # Parse the HTML content
        print("Parsing product data...")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Initialize product details dictionary
        product_details = {
            "title": None,
            "price": None,
            "currency": None,
            "description": None,
            "features": [],
            "specifications": {},
            "rating": None,
            "review_count": None,
            "image_urls": [],
            "availability": None,
            "seller": None,
            "brand": None,
            "categories": [],
            "url": url
        }
        
        # Extract product title
        title_element = soup.select_one('#productTitle')
        if title_element:
            product_details["title"] = title_element.text.strip()
        
        # Extract product price
        price_whole = soup.select_one('.a-price-whole')
        price_fraction = soup.select_one('.a-price-fraction')
        if price_whole and price_fraction:
            product_details["price"] = f"{price_whole.text.strip()}{price_fraction.text.strip()}"
            product_details["currency"] = soup.select_one('.a-price-symbol').text.strip() if soup.select_one('.a-price-symbol') else "$"
        else:
            price_element = soup.select_one('.a-offscreen')
            if price_element:
                price_text = price_element.text.strip()
                # Extract currency symbol
                currency_match = re.search(r'[^\d\.,]+', price_text)
                if currency_match:
                    product_details["currency"] = currency_match.group(0).strip()
                # Extract price value
                price_match = re.search(r'[\d\.,]+', price_text)
                if price_match:
                    product_details["price"] = price_match.group(0).strip()
        
        # Extract product description
        description_element = soup.select_one('#productDescription')
        if description_element:
            product_details["description"] = description_element.text.strip()
        
        # Extract product features
        feature_elements = soup.select('#feature-bullets ul li:not(.aok-hidden) span.a-list-item')
        if feature_elements:
            product_details["features"] = [feature.text.strip() for feature in feature_elements]
        
        # Extract product specifications
        spec_elements = soup.select('#productDetails_techSpec_section_1 tr, #productDetails_detailBullets_sections1 tr')
        for spec in spec_elements:
            key_element = spec.select_one('th, .prodDetSectionEntry')
            value_element = spec.select_one('td, .prodDetAttrValue')
            if key_element and value_element:
                key = key_element.text.strip().rstrip(':')
                value = value_element.text.strip()
                product_details["specifications"][key] = value
        
        # Extract product rating
        rating_element = soup.select_one('#acrPopover')
        if rating_element and 'title' in rating_element.attrs:
            rating_text = rating_element['title']
            rating_match = re.search(r'([\d\.]+)', rating_text)
            if rating_match:
                product_details["rating"] = float(rating_match.group(1))
        
        # Extract review count
        review_count_element = soup.select_one('#acrCustomerReviewText')
        if review_count_element:
            count_text = review_count_element.text.strip()
            count_match = re.search(r'([\d,]+)', count_text)
            if count_match:
                product_details["review_count"] = int(count_match.group(1).replace(',', ''))
        
        # Extract product images
        # Try to get from image gallery first
        image_gallery = soup.select('#altImages .item')
        for img_item in image_gallery:
            img_element = img_item.select_one('img')
            if img_element and 'src' in img_element.attrs:
                img_url = img_element['src']
                # Convert thumbnail URL to full-size image URL
                full_img_url = re.sub(r'\._.*_\.', '.', img_url)
                if full_img_url not in product_details["image_urls"]:
                    product_details["image_urls"].append(full_img_url)
        
        # If no images found in gallery, try main product image
        if not product_details["image_urls"]:
            main_img = soup.select_one('#landingImage, #imgBlkFront')
            if main_img and 'data-old-hires' in main_img.attrs:
                product_details["image_urls"].append(main_img['data-old-hires'])
            elif main_img and 'src' in main_img.attrs:
                product_details["image_urls"].append(main_img['src'])
        
        # Extract product availability
        availability_element = soup.select_one('#availability')
        if availability_element:
            product_details["availability"] = availability_element.text.strip()
        
        # Extract seller information
        seller_element = soup.select_one('#merchant-info')
        if seller_element:
            product_details["seller"] = seller_element.text.strip()
        
        # Extract brand information
        brand_element = soup.select_one('#bylineInfo')
        if brand_element:
            product_details["brand"] = brand_element.text.strip()
        
        # Extract category information
        category_elements = soup.select('#wayfinding-breadcrumbs_feature_div ul li')
        if category_elements:
            product_details["categories"] = [category.text.strip() for category in category_elements if category.text.strip()]
        
        return product_details
        
    except Exception as e:
        return {"error": f"Error fetching product details: {str(e)}"}


def download_image(url: str) -> Optional[Image.Image]:
    """
    Download an image from a URL
    
    Args:
        url (str): URL of the image
        
    Returns:
        Optional[Image.Image]: PIL Image object or None if download failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        return None
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return None


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image to base64 string
    
    Args:
        image_path (str): Path to the image or URL
        
    Returns:
        Optional[str]: Base64 encoded string or None if encoding failed
    """
    try:
        if image_path.startswith('http'):
            response = requests.get(image_path, stream=True)
            if response.status_code == 200:
                image_data = response.content
            else:
                return None
        else:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None


def generate_product_demo_script(product_details: Dict[str, Any]) -> str:
    """
    Generate a product demo video script using OpenAI
    
    Args:
        product_details (Dict[str, Any]): Product details dictionary
        
    Returns:
        str: Generated script for the product demo video
    """
    # Prepare the product information for the prompt
    product_info = {
        "title": product_details.get("title", "Unknown Product"),
        "description": product_details.get("description", ""),
        "features": product_details.get("features", []),
        "specifications": product_details.get("specifications", {}),
        "price": f"{product_details.get('currency', '$')}{product_details.get('price', 'N/A')}",
        "rating": product_details.get("rating", "N/A"),
        "brand": product_details.get("brand", "Unknown Brand")
    }
    
    logger.info(f"Generating product demo script for: {product_info['title']}")
    
    # Create a prompt for OpenAI
    prompt = f"""
    You are a professional product marketing specialist. Analyze the following product information and create a compelling 30-40 second product demo video script.
    
    PRODUCT INFORMATION:
    Title: {product_info['title']}
    Brand: {product_info['brand']}
    Price: {product_info['price']}
    Rating: {product_info['rating']}
    
    Description:
    {product_info['description']}
    
    Key Features:
    {', '.join(product_info['features']) if product_info['features'] else 'No features listed'}
    
    Specifications:
    {json.dumps(product_info['specifications'], indent=2)}
    
    Create a script that highlights the product's key benefits, features, and use cases. The script should be engaging, informative, and persuasive.
    Structure the script with clear sections, including an introduction, key features, benefits, and a call to action.
    The script should be approximately 30-40 seconds when read aloud at a normal pace.
    """
    
    try:
        logger.info("Sending request to OpenAI for script generation")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional product marketing specialist who creates compelling product demo scripts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        script = response.choices[0].message.content.strip()
        logger.info(f"Generated script ({len(script)} chars):\n{script}\n")
        return script
    except Exception as e:
        error_msg = f"Error generating product demo script: {str(e)}"
        logger.error(error_msg)
        return f"Failed to generate script: {str(e)}"


def generate_creative_prompts(script: str, image_url: str) -> Dict[str, Any]:
    """
    Generate creative prompts for image and video content based on the script and source image
    
    Args:
        script (str): The product demo script
        image_url (str): URL of the product image
        
    Returns:
        Dict[str, Any]: Dictionary containing image prompt, video prompt, and image URL
    """
    logger.info(f"Generating creative prompts for image: {image_url}")
    
    # Encode the image to base64 if it's a URL
    image_base64 = encode_image_to_base64(image_url)
    
    if not image_base64:
        error_msg = "Failed to encode image to base64"
        logger.error(error_msg)
        return {
            "error": "Failed to encode image",
            "img_prompt": "",
            "video_prompt": "",
            "img_url": image_url
        }
    
    # Parse the script to extract scenes
    scenes = extract_scenes_from_script(script)
    
    # Create a prompt for OpenAI
    prompt = f"""
    You are tasked with generating creative prompts for both image and video content based on a provided script and source images. Follow the instructions below:

    The script has been divided into the following scenes:
    {json.dumps(scenes, indent=2)}
    
    For each scene, create:
    1. An image prompt that captures the essence of that scene using the provided product image
    2. A video prompt that describes how to animate that scene
    
    Image Prompt Objective:
    Generate an image prompt for each scene based on the existing product image. The output image should be visually compelling and designed to match the scene's content. Ensure the image aligns with the narrative or mood of the provided script scene.

    Video Prompt Objective:
    Using the image prompt and the product image, write a corresponding video prompt that describes how the image could be brought to life through motion (e.g., camera panning, lighting changes, environmental motion). The video prompt should be based on the same theme or story as the script scene.

    Human Interaction Rule:
    You may include human figures and interactions in the image or video concepts, but faces must not be visible. Use poses, silhouettes, or over-the-shoulder angles to preserve anonymity.

    PRODUCT IMAGE URL:
    {image_url}
    
    Respond in JSON format with the following structure:
    {{
      "scenes": [
        {{
          "scene_name": "Scene name/number",
          "scene_content": "Original scene content",
          "img_prompt": "The prompt for generating an enhanced product image for this scene",
          "video_prompt": "The prompt for animating this scene into a video"
        }},
        // Additional scenes...
      ]
    }}
    """
    
    logger.info("Sending request to OpenAI for creative prompts generation")
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a creative director specializing in product marketing visuals. You create compelling image and video prompts based on product scripts and images."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result_text = response.choices[0].message.content
        logger.info(f"Generated creative prompts:\n{result_text}\n")
        
        result = json.loads(result_text)
        
        # Add the original image URL to each scene
        if "scenes" in result:
            for scene in result["scenes"]:
                scene["img_url"] = image_url
        
        # Add the original image URL to the top level for backward compatibility
        result["img_url"] = image_url
        
        return result
    except Exception as e:
        error_msg = f"Error generating creative prompts: {str(e)}"
        logger.error(error_msg)
        return {
            "error": f"Failed to generate prompts: {str(e)}",
            "img_prompt": "",
            "video_prompt": "",
            "img_url": image_url
        }


def extract_scenes_from_script(script: str) -> List[Dict[str, str]]:
    """
    Extract scenes from a script based on section markers
    
    Args:
        script (str): The product demo script
        
    Returns:
        List[Dict[str, str]]: List of scenes with name and content
    """
    # Look for scene markers like [INTRO], [KEY FEATURES], etc.
    scene_pattern = r'\*\*\[(.*?)\].*?\*\*\s*(.*?)(?=\*\*\[|$)'  # Matches **[SCENE_NAME]** and content until next scene or end
    scenes = []
    
    # Try to find scenes with the pattern
    matches = re.findall(scene_pattern, script, re.DOTALL)
    
    if matches:
        for i, (scene_name, scene_content) in enumerate(matches):
            scenes.append({
                "scene_number": i + 1,
                "scene_name": scene_name.strip(),
                "scene_content": scene_content.strip()
            })
    else:
        # If no scenes found, split by paragraphs or use the whole script as one scene
        paragraphs = script.split('\n\n')
        if len(paragraphs) > 1:
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    scenes.append({
                        "scene_number": i + 1,
                        "scene_name": f"Scene {i+1}",
                        "scene_content": paragraph.strip()
                    })
        else:
            scenes.append({
                "scene_number": 1,
                "scene_name": "Main Scene",
                "scene_content": script.strip()
            })
    
    return scenes


def generate_enhanced_image(img_prompt: str, reference_image_url: str, output_dir: str = "output") -> Optional[str]:
    """
    Generate an enhanced image based on the prompt and reference image
    
    Args:
        img_prompt (str): The image generation prompt
        reference_image_url (str): URL of the reference image
        output_dir (str): Directory to save the generated image
        
    Returns:
        Optional[str]: Path to the generated image or None if generation failed
    """
    logger.info(f"Generating enhanced image using prompt: {img_prompt[:100]}...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the reference image directly
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(reference_image_url, headers=headers, stream=True, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to download reference image: HTTP {response.status_code}")
            return None
            
        # Save the image to a temporary file
        temp_image_path = os.path.join(output_dir, f"temp_reference_{uuid.uuid4()}.png")
        img = Image.open(BytesIO(response.content))
        img.save(temp_image_path, format="PNG")
        logger.info(f"Saved reference image to temporary file: {temp_image_path}")
        
        # Generate a new image using OpenAI's API
        logger.info("Generating new image with DALL-E")
        
        # Use generate instead of edit since edit is having format issues
        with open(temp_image_path, "rb") as img_file:
            response = openai.images.generate(
                model="dall-e-2",
                prompt=f"Based on this product image, create an enhanced version that: {img_prompt}. Keep the original product shape and design unchanged.",
                n=1,
                size="1024x1024"
            )
        
        # Get the image URL from the response
        image_url = response.data[0].url
        logger.info(f"Generated image URL: {image_url}")
        
        # Download the generated image
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # Generate a unique filename
            filename = f"{uuid.uuid4()}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Save the image
            with open(output_path, "wb") as f:
                f.write(image_response.content)
            
            logger.info(f"Saved generated image to: {output_path}")
            
            # Clean up temporary file
            try:
                os.remove(temp_image_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
                
            return output_path
        else:
            logger.error(f"Failed to download generated image: HTTP {image_response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error generating enhanced image: {str(e)}")
        return None


def process_product(url: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    Process a product URL to generate enhanced content
    
    Args:
        url (str): Amazon product URL
        output_dir (str): Directory to save generated content
        
    Returns:
        Dict[str, Any]: Results of the processing
    """
    logger.info(f"Starting to process product URL: {url}")
    results = {
        "product_details": None,
        "demo_script": None,
        "creative_prompts": None,
        "generated_images": []
    }
    
    # Step 1: Fetch product details using the more reliable bs4scrape function
    logger.info("Step 1: Fetching product details...")
    bs4_product_data = bs4_get_amazon_product_details(url)
    
    if "error" in bs4_product_data:
        error_msg = f"Error fetching product details: {bs4_product_data['error']}"
        logger.error(error_msg)
        results["error"] = bs4_product_data["error"]
        return results
    
    # Convert bs4scrape format to our internal format
    product_details = {
        "title": bs4_product_data.get("title"),
        "price": bs4_product_data.get("price"),
        "currency": bs4_product_data.get("price", "$")[0] if bs4_product_data.get("price") else "$",
        "description": bs4_product_data.get("product_description", ""),
        "features": [],  # Will be populated from description if available
        "specifications": bs4_product_data.get("product_details", {}),
        "rating": bs4_product_data.get("rating"),
        "review_count": bs4_product_data.get("number_of_reviews"),
        "image_urls": bs4_product_data.get("images", []),
        "availability": bs4_product_data.get("availability"),
        "brand": bs4_product_data.get("brand"),
        "categories": [],
        "url": url
    }
    
    # Extract features from description if available
    if product_details["description"]:
        # Try to extract bullet points or key features from description
        features = re.findall(r'[•\*\-]\s*([^\n]+)', product_details["description"])
        if features:
            product_details["features"] = features
    
    logger.info(f"Successfully fetched details for product: {product_details.get('title', 'Unknown')}")
    logger.info(f"Found {len(product_details['image_urls'])} product images")
    results["product_details"] = product_details
    
    # Step 2: Generate product demo script
    logger.info("Step 2: Generating product demo script...")
    demo_script = generate_product_demo_script(product_details)
    results["demo_script"] = demo_script
    
    # Step 3: Generate creative prompts for each image
    logger.info("Step 3: Generating creative prompts...")
    creative_prompts_list = []
    
    # Use only the first image for simplicity, or loop through all images if needed
    if product_details.get("image_urls"):
        for i, image_url in enumerate(product_details["image_urls"][:1]):  # Limit to first image for now
            logger.info(f"Processing image {i+1}/{len(product_details['image_urls'][:1])}: {image_url}")
            creative_prompts = generate_creative_prompts(demo_script, image_url)
            creative_prompts_list.append(creative_prompts)
    else:
        logger.warning("No product images found in the product details")
    
    results["creative_prompts"] = creative_prompts_list
    
    # Step 4: Generate enhanced images
    logger.info("Step 4: Generating enhanced images...")
    for i, prompts in enumerate(creative_prompts_list):
        if "error" not in prompts:
            logger.info(f"Generating enhanced image {i+1}/{len(creative_prompts_list)}...")
            for scene in prompts["scenes"]:
                img_path = generate_enhanced_image(
                    scene["img_prompt"],
                    prompts["img_url"],
                    output_dir
                )
                if img_path:
                    logger.info(f"Successfully generated enhanced image: {img_path}")
                    results["generated_images"].append({
                        "original_image": prompts["img_url"],
                        "enhanced_image": img_path,
                        "img_prompt": scene["img_prompt"],
                        "video_prompt": scene["video_prompt"]
                    })
                else:
                    logger.error(f"Failed to generate enhanced image for prompt {i+1}")
        else:
            logger.error(f"Skipping image generation due to error in prompts: {prompts.get('error', 'Unknown error')}")
    
    logger.info("Product processing complete")
    return results


def main():
    parser = argparse.ArgumentParser(description='Extract product details and generate enhanced content from Amazon URLs')
    parser.add_argument('url', help='Amazon product URL')
    parser.add_argument('--output', '-o', help='Output directory for generated content', default='output')
    parser.add_argument('--json-output', '-j', help='Output file path for JSON results', default=None)
    parser.add_argument('--verbose', '-v', action='store_true', help='Display detailed information including prompts')
    args = parser.parse_args()
    
    # Validate URL - check for both amazon.com and shortened amzn links
    if not ('amazon.' in args.url or 'amzn.' in args.url):
        print("Error: The URL doesn't appear to be from Amazon.")
        sys.exit(1)
    
    # Process the product
    results = process_product(args.url, args.output)
    
    # Print a summary of the results
    print("\n" + "=" * 80)
    print(f"PRODUCT: {results.get('product_details', {}).get('title', 'Unknown')}")
    print("=" * 80)
    
    # Print script
    print("\nGENERATED SCRIPT:")
    print("-" * 80)
    script = results.get('demo_script', '')
    print(f"{script[:300]}..." if len(script) > 300 else script)
    print(f"\nTotal script length: {len(script)} characters")
    
    # Print creative prompts
    if args.verbose:
        print("\nCREATIVE PROMPTS:")
        print("-" * 80)
        for i, prompts in enumerate(results.get('creative_prompts', [])):
            print(f"\nPrompt Set #{i+1}:")
            if 'error' in prompts:
                print(f"Error: {prompts['error']}")
            else:
                for scene in prompts["scenes"]:
                    print(f"\nScene: {scene['scene_name']}")
                    print(f"Image Prompt:\n{scene.get('img_prompt', 'None')}")
                    print(f"Video Prompt:\n{scene.get('video_prompt', 'None')}")
                    print(f"Reference Image URL: {prompts.get('img_url', 'None')}")
    else:
        print(f"\nCreative prompts generated: {len(results.get('creative_prompts', []))}")
    
    # Print generated images
    print("\nGENERATED IMAGES:")
    print("-" * 80)
    if results.get("generated_images"):
        for i, img_data in enumerate(results["generated_images"]):
            print(f"  {i+1}. {img_data['enhanced_image']}")
            if args.verbose:
                print(f"     Original: {img_data['original_image']}")
                print(f"     Prompt: {img_data['img_prompt'][:100]}..." if len(img_data['img_prompt']) > 100 else img_data['img_prompt'])
    else:
        print("  No images were generated.")
    
    # Save results to JSON file if specified
    if args.json_output:
        try:
            # Create a serializable version of the results
            serializable_results = {
                "product_details": results.get("product_details"),
                "demo_script": results.get("demo_script"),
                "creative_prompts": results.get("creative_prompts"),
                "generated_images": [
                    {
                        "original_image": img_data["original_image"],
                        "enhanced_image": img_data["enhanced_image"],
                        "img_prompt": img_data["img_prompt"],
                        "video_prompt": img_data["video_prompt"]
                    } for img_data in results.get("generated_images", [])
                ]
            }
            
            with open(args.json_output, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.json_output}")
        except Exception as e:
            print(f"\nError saving to file: {str(e)}")
    
    print("\nLog file with detailed information has been saved to: scrapeer.log")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)
