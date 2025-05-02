#!/usr/bin/env python3
"""
Amazon Product Automated Content Review
----------------------------------
A script to scrape Amazon product details and automatically generate content reviews.

Usage:
    python scrapeer.py <amazon_product_url> [--output filename.json] [--review-type detailed|summary|comparison] [--generate-image] [--image-output-dir directory] [--generate-video] [--video-duration seconds] [--with-voiceover]

Example:
    python scrapeer.py https://www.amazon.com/dp/B07PXGQC1Q --output product_review.json --review-type detailed --generate-image --image-output-dir images --generate-video --video-duration 90 --with-voiceover
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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches

# Load environment variables from .env file
load_dotenv()

# Check if OpenAI API key is set in environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def fetch_amazon_product_details(url: str) -> Dict[str, Any]:
    """
    Fetches product details from an Amazon product URL
    
    Args:
        url (str): Amazon product URL
        
    Returns:
        dict: Product details including title, price, description, features, and reviews
    """
    # List of user agents to rotate
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1'
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
            "top_reviews": [],
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
        currency_symbol = soup.select_one('.a-price-symbol')
        
        if price_whole and price_fraction:
            product_details["price"] = f"{price_whole.text.strip()}{price_fraction.text.strip()}"
            if currency_symbol:
                product_details["currency"] = currency_symbol.text.strip()
        else:
            # Alternative price selectors
            price_element = soup.select_one('.a-price .a-offscreen')
            if price_element:
                price_text = price_element.text.strip()
                # Extract currency symbol
                currency_match = re.search(r'^(\D+)', price_text)
                if currency_match:
                    product_details["currency"] = currency_match.group(1)
                # Extract price
                price_match = re.search(r'[\d,.]+', price_text)
                if price_match:
                    product_details["price"] = price_match.group(0)
        
        # Extract product description
        description_element = soup.select_one('#productDescription')
        if description_element:
            product_details["description"] = description_element.text.strip()
        else:
            # Alternative description selectors
            alt_desc = soup.select_one('#feature-bullets')
            if alt_desc:
                product_details["description"] = alt_desc.text.strip()
        
        # Extract product features
        feature_bullets = soup.select('#feature-bullets ul li')
        for bullet in feature_bullets:
            # Skip promotional elements
            if 'promotions_feature_div' not in bullet.get('id', ''):
                feature_text = bullet.text.strip()
                if feature_text:
                    product_details["features"].append(feature_text)
        
        # Extract product specifications
        spec_tables = soup.select('.prodDetTable')
        for table in spec_tables:
            rows = table.select('tr')
            for row in rows:
                header = row.select_one('th')
                value = row.select_one('td')
                if header and value:
                    header_text = header.text.strip().rstrip(':')
                    value_text = value.text.strip()
                    if header_text and value_text:
                        product_details["specifications"][header_text] = value_text
        
        # Extract rating
        rating_element = soup.select_one('#acrPopover')
        if rating_element and 'title' in rating_element.attrs:
            rating_text = rating_element['title']
            rating_match = re.search(r'([\d.]+)', rating_text)
            if rating_match:
                product_details["rating"] = rating_match.group(1)
        
        # Extract review count
        review_count_element = soup.select_one('#acrCustomerReviewText')
        if review_count_element:
            count_text = review_count_element.text.strip()
            count_match = re.search(r'([\d,]+)', count_text)
            if count_match:
                product_details["review_count"] = count_match.group(1).replace(',', '')
        
        # Extract top reviews
        reviews = soup.select('#cm-cr-dp-review-list .review')
        for review in reviews[:5]:  # Get top 5 reviews
            review_data = {
                "rating": None,
                "title": None,
                "author": None,
                "date": None,
                "text": None,
                "verified": False
            }
            
            # Review rating
            rating_element = review.select_one('.review-rating')
            if rating_element:
                rating_text = rating_element.text.strip()
                rating_match = re.search(r'([\d.]+)', rating_text)
                if rating_match:
                    review_data["rating"] = rating_match.group(1)
            
            # Review title
            title_element = review.select_one('.review-title')
            if title_element:
                review_data["title"] = title_element.text.strip()
            
            # Review author
            author_element = review.select_one('.a-profile-name')
            if author_element:
                review_data["author"] = author_element.text.strip()
            
            # Review date
            date_element = review.select_one('.review-date')
            if date_element:
                review_data["date"] = date_element.text.strip()
            
            # Review text
            text_element = review.select_one('.review-text')
            if text_element:
                review_data["text"] = text_element.text.strip()
            
            # Verified purchase
            verified_element = review.select_one('.a-color-state')
            if verified_element and 'verified purchase' in verified_element.text.lower():
                review_data["verified"] = True
            
            product_details["top_reviews"].append(review_data)
        
        # Extract image URLs
        image_gallery = soup.select('#altImages .item')
        for item in image_gallery:
            img = item.select_one('img')
            if img and 'src' in img.attrs:
                img_url = img['src']
                # Convert thumbnail URL to full-size image URL
                img_url = re.sub(r'\._.*_\.', '.', img_url)
                product_details["image_urls"].append(img_url)
        
        # Extract availability
        availability_element = soup.select_one('#availability')
        if availability_element:
            product_details["availability"] = availability_element.text.strip()
        
        # Extract seller
        seller_element = soup.select_one('#merchant-info')
        if seller_element:
            product_details["seller"] = seller_element.text.strip()
        
        # Extract brand
        brand_element = soup.select_one('#bylineInfo')
        if brand_element:
            brand_text = brand_element.text.strip()
            brand_match = re.search(r'by\s+(.+)', brand_text, re.IGNORECASE)
            if brand_match:
                product_details["brand"] = brand_match.group(1).strip()
        
        # Extract categories
        breadcrumbs = soup.select('#wayfinding-breadcrumbs_feature_div ul li')
        for crumb in breadcrumbs:
            link = crumb.select_one('a')
            if link:
                category = link.text.strip()
                if category:
                    product_details["categories"].append(category)
        
        return product_details
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def generate_product_review(product_details: Dict[str, Any], review_type: str = "detailed") -> Dict[str, Any]:
    """
    Generate an automated review for a product based on its details
    
    Args:
        product_details (dict): Product details dictionary
        review_type (str): Type of review to generate (detailed, summary, comparison)
        
    Returns:
        dict: Generated review content
    """
    if "error" in product_details:
        return {"error": product_details["error"]}
    
    # Prepare review data structure
    review = {
        "product_title": product_details.get("title", "Unknown Product"),
        "review_type": review_type,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "content": {}
    }
    
    # Check if we have OpenAI API key for AI-generated content
    if OPENAI_API_KEY:
        try:
            print(f"Generating {review_type} review using AI...")
            review["content"] = generate_ai_review(product_details, review_type)
        except Exception as e:
            print(f"Error generating AI review: {str(e)}")
            # Fall back to template-based review
            review["content"] = generate_template_review(product_details, review_type)
    else:
        print("No OpenAI API key found. Using template-based review generation.")
        review["content"] = generate_template_review(product_details, review_type)
    
    return review

def generate_ai_review(product_details: Dict[str, Any], review_type: str) -> Dict[str, str]:
    """
    Generate a review using OpenAI's API
    
    Args:
        product_details (dict): Product details dictionary
        review_type (str): Type of review to generate
        
    Returns:
        dict: Generated review content sections
    """
    # Configure OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Prepare product information for the prompt
    product_info = f"Product: {product_details.get('title', 'Unknown Product')}\n"
    product_info += f"Brand: {product_details.get('brand', 'Unknown')}\n"
    product_info += f"Price: {product_details.get('currency', '$')}{product_details.get('price', 'N/A')}\n"
    product_info += f"Rating: {product_details.get('rating', 'N/A')}/5 from {product_details.get('review_count', 'N/A')} reviews\n"
    
    if product_details.get("description"):
        product_info += f"\nDescription: {product_details['description']}\n"
    
    if product_details.get("features"):
        product_info += "\nFeatures:\n"
        for feature in product_details["features"]:
            product_info += f"- {feature}\n"
    
    if product_details.get("specifications"):
        product_info += "\nSpecifications:\n"
        for key, value in product_details["specifications"].items():
            product_info += f"- {key}: {value}\n"
    
    if product_details.get("top_reviews"):
        product_info += "\nCustomer Reviews:\n"
        for i, review in enumerate(product_details["top_reviews"][:3], 1):
            product_info += f"Review {i}: {review.get('rating', 'N/A')}/5 - {review.get('title', 'N/A')}\n"
            product_info += f"{review.get('text', 'N/A')}\n\n"
    
    # Define prompts based on review type
    if review_type == "detailed":
        prompt = f"You are a professional product reviewer. Create a detailed, comprehensive review for the following Amazon product:\n\n{product_info}\n\nYour review should include:\n1. An engaging introduction\n2. Pros and cons analysis\n3. Feature evaluation\n4. Value for money assessment\n5. Comparison with similar products (if information available)\n6. Final verdict with rating out of 10\n7. Recommendation for who should buy this product"
    
    elif review_type == "summary":
        prompt = f"You are a concise product reviewer. Create a brief, informative summary review for the following Amazon product:\n\n{product_info}\n\nYour review should include:\n1. A short introduction\n2. Key highlights (3-5 points)\n3. Main drawbacks (if any)\n4. Quick verdict with rating out of 10"
    
    elif review_type == "comparison":
        prompt = f"You are a comparative product analyst. Create a review that positions this product in the market context:\n\n{product_info}\n\nYour review should include:\n1. Market positioning\n2. Competitive advantages and disadvantages\n3. Value proposition analysis\n4. Alternative recommendations for different user needs\n5. Final verdict on who should choose this product over alternatives"
    
    else:  # Default to balanced review
        prompt = f"You are a balanced product reviewer. Create a fair, unbiased review for the following Amazon product:\n\n{product_info}\n\nYour review should include:\n1. Product overview\n2. Equal coverage of strengths and weaknesses\n3. Practical use cases\n4. Value assessment\n5. Final recommendation with rating out of 10"
    
    # Make API call to OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert product reviewer who writes engaging, informative, and honest reviews."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1500
    )
    
    # Process the response
    review_text = response.choices[0].message.content
    
    # Parse the review into sections
    sections = {}
    
    # Extract introduction (first paragraph)
    intro_match = re.search(r'^(.+?)\n\n', review_text, re.DOTALL)
    if intro_match:
        sections["introduction"] = intro_match.group(1).strip()
    
    # Extract pros and cons if present
    pros_match = re.search(r'(?:Pros|PROS|Advantages|Strengths):\s*(.+?)(?:\n\n|\n(?:Cons|CONS|Disadvantages|Weaknesses):)', review_text, re.DOTALL)
    if pros_match:
        sections["pros"] = pros_match.group(1).strip()
    
    cons_match = re.search(r'(?:Cons|CONS|Disadvantages|Weaknesses):\s*(.+?)\n\n', review_text, re.DOTALL)
    if cons_match:
        sections["cons"] = cons_match.group(1).strip()
    
    # Extract verdict/conclusion (usually last paragraph)
    verdict_match = re.search(r'(?:Verdict|Conclusion|Final Thoughts|Bottom Line|Summary):\s*(.+?)$', review_text, re.DOTALL)
    if verdict_match:
        sections["verdict"] = verdict_match.group(1).strip()
    else:
        # If no explicit verdict section, use the last paragraph
        paragraphs = review_text.split('\n\n')
        if paragraphs:
            sections["verdict"] = paragraphs[-1].strip()
    
    # Extract rating if present
    rating_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of|\/)\s*10', review_text)
    if rating_match:
        sections["rating"] = rating_match.group(1)
    
    # Include the full review text
    sections["full_text"] = review_text
    
    return sections

def generate_template_review(product_details: Dict[str, Any], review_type: str) -> Dict[str, str]:
    """
    Generate a review using templates
    
    Args:
        product_details (dict): Product details dictionary
        review_type (str): Type of review to generate
        
    Returns:
        dict: Generated review content sections
    """
    review_content = {}
    
    # Get product information
    title = product_details.get("title", "this product")
    brand = product_details.get("brand", "the manufacturer")
    price = product_details.get("price", "N/A")
    currency = product_details.get("currency", "$")
    rating = product_details.get("rating", "N/A")
    features = product_details.get("features", [])
    
    # Generate introduction
    intro_templates = [
        f"The {title} by {brand} is a product that has gained attention in the market.",
        f"We recently had the opportunity to evaluate the {title} from {brand}.",
        f"In today's review, we'll be taking a close look at the {title}, a product from {brand}."
    ]
    review_content["introduction"] = random.choice(intro_templates)
    
    # Generate pros
    pros = []
    if features:
        # Use some features as pros
        for feature in features[:min(3, len(features))]:
            pros.append(feature)
    
    if float(rating) >= 4.0 if rating != "N/A" else False:
        pros.append(f"High customer satisfaction with an average rating of {rating}/5")
    
    if not pros:
        pros = ["Appears to meet basic functionality requirements", "Available for purchase online"]
    
    review_content["pros"] = "\n- " + "\n- ".join(pros)
    
    # Generate cons
    cons = []
    if float(rating) < 4.0 if rating != "N/A" else False:
        cons.append(f"Customer ratings indicate room for improvement with an average of {rating}/5")
    
    if not cons:
        cons = ["Limited information available for comprehensive assessment", "May not meet all specialized requirements"]
    
    review_content["cons"] = "\n- " + "\n- ".join(cons)
    
    # Generate verdict
    if review_type == "detailed":
        verdict_template = f"After thorough evaluation, the {title} offers {len(pros)} notable advantages including {pros[0].lower() if pros else 'basic functionality'}. However, potential buyers should consider {cons[0].lower() if cons else 'their specific needs before purchasing'}. Overall, it represents a {random.choice(['reasonable', 'considerable', 'noteworthy'])} option in its category."
    elif review_type == "summary":
        verdict_template = f"In summary, the {title} is a {random.choice(['suitable', 'functional', 'practical'])} product with some {random.choice(['strengths', 'benefits', 'advantages'])} that may meet the needs of certain users."
    else:  # comparison or default
        verdict_template = f"Compared to alternatives in the market, the {title} positions itself as a {random.choice(['contender', 'option', 'alternative'])} worth considering for those who prioritize {pros[0].lower() if pros else 'the features it offers'}."
    
    review_content["verdict"] = verdict_template
    
    # Generate rating
    if rating != "N/A":
        # Convert 5-star rating to 10-point scale
        rating_10pt = float(rating) * 2
        # Add some randomness but keep it reasonable based on original rating
        rating_10pt = max(1, min(10, rating_10pt + random.uniform(-0.5, 0.5)))
        review_content["rating"] = f"{rating_10pt:.1f}"
    else:
        review_content["rating"] = f"{random.uniform(5.0, 8.0):.1f}"
    
    # Combine all sections for full text
    full_text = review_content["introduction"] + "\n\n"
    full_text += "Pros:\n" + review_content["pros"] + "\n\n"
    full_text += "Cons:\n" + review_content["cons"] + "\n\n"
    full_text += "Verdict:\n" + review_content["verdict"] + "\n\n"
    full_text += f"Rating: {review_content['rating']}/10"
    
    review_content["full_text"] = full_text
    
    return review_content

def download_image(url: str) -> Optional[Image.Image]:
    """
    Download an image from a URL
    
    Args:
        url (str): URL of the image
        
    Returns:
        Optional[Image.Image]: PIL Image object or None if download failed
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f"Failed to download image: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return None

def analyze_image_with_ai(image: Image.Image, product_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze product image using OpenAI's vision capabilities
    
    Args:
        image (Image.Image): PIL Image object
        product_details (Dict[str, Any]): Product details
        
    Returns:
        Dict[str, Any]: Analysis results with feature annotations
    """
    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Prepare product information for context
    product_info = f"Product: {product_details.get('title', 'Unknown Product')}\n"
    product_info += f"Brand: {product_details.get('brand', 'Unknown')}\n"
    
    if product_details.get("features"):
        product_info += "\nFeatures:\n"
        for feature in product_details["features"][:5]:  # Limit to top 5 features
            product_info += f"- {feature}\n"
    
    if product_details.get("specifications"):
        product_info += "\nSpecifications:\n"
        for key, value in list(product_details["specifications"].items())[:5]:  # Limit to top 5 specs
            product_info += f"- {key}: {value}\n"
    
    # Configure OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Create prompt for image analysis
    prompt = f"""You are an expert product analyst. Analyze this product image and identify key visual features that should be highlighted.

Product Information:
{product_info}

For this image:
1. Identify 3-5 key visual features worth highlighting (e.g., screen size for a laptop, camera setup for a phone)
2. For each feature, provide:
   - A short name (1-3 words)
   - A brief description (5-15 words)
   - The approximate location in the image (top-left, center, bottom-right, etc.)
   - A confidence score (0-100%)

Format your response as a JSON object with an array of features.
"""
    
    try:
        # Make API call to OpenAI with vision capabilities
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o which has vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Extract and parse the response
        analysis_text = response.choices[0].message.content
        
        # Try to extract JSON from the response
        try:
            # Find JSON object in the response
            json_match = re.search(r'\{[\s\S]*\}', analysis_text)
            if json_match:
                analysis_json = json.loads(json_match.group(0))
                return analysis_json
            else:
                # If no JSON found, create a structured response
                return {"features": [{"name": "Product Overview", "description": "Key product features and specifications", "location": "center", "confidence": 90}]}
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response
            return {"features": [{"name": "Product Overview", "description": "Key product features and specifications", "location": "center", "confidence": 90}]}
    
    except Exception as e:
        print(f"Error analyzing image with AI: {str(e)}")
        # Return a default analysis
        return {"features": [{"name": "Product Overview", "description": "Key product features and specifications", "location": "center", "confidence": 90}]}

def process_product_images(product_details: Dict[str, Any], output_dir: str = "output") -> List[str]:
    """
    Process product images to generate multiple annotated feature highlights
    
    Args:
        product_details (Dict[str, Any]): Product details including image URLs
        output_dir (str): Directory to save generated images
        
    Returns:
        List[str]: Paths to the generated images
    """
    if not product_details.get("image_urls"):
        print("No product images found")
        return []
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get product title for filenames
    product_title = product_details.get("title", "product")
    safe_title = re.sub(r'[^\w\s-]', '', product_title)
    safe_title = re.sub(r'[-\s]+', '-', safe_title).strip('-').lower()
    
    # List to store paths of generated images
    generated_image_paths = []
    
    # Process up to 3 product images if available
    image_urls = product_details["image_urls"][:min(3, len(product_details["image_urls"]))]
    
    for i, image_url in enumerate(image_urls):
        print(f"Processing image {i+1}/{len(image_urls)}: {image_url}")
        
        # Download the image
        image = download_image(image_url)
        if not image:
            print(f"Failed to download product image: {image_url}")
            continue
        
        # Analyze the image with AI
        print(f"Analyzing image {i+1} with AI...")
        analysis = analyze_image_with_ai(image, product_details)
        
        # Generate annotated image
        print(f"Generating annotated image {i+1}...")
        annotated_image_path = generate_annotated_image(image, analysis, product_details, output_dir, i+1)
        
        if annotated_image_path:
            generated_image_paths.append(annotated_image_path)
            
            # Generate a realistic review scene for this image
            print(f"Creating realistic review scene for image {i+1}...")
            scene_image_path = generate_review_scene(image, product_details, output_dir, i+1)
            if scene_image_path:
                generated_image_paths.append(scene_image_path)
    
    return generated_image_paths

def generate_annotated_image(image: Image.Image, analysis: Dict[str, Any], product_details: Dict[str, Any], output_path: str, image_index: int = 1) -> Optional[str]:
    """
    Generate an annotated image highlighting product features
    
    Args:
        image (Image.Image): Original product image
        analysis (Dict[str, Any]): Image analysis results
        product_details (Dict[str, Any]): Product details
        output_path (str): Directory to save the generated image
        image_index (int): Index of the image being processed
        
    Returns:
        Optional[str]: Path to the generated image or None if generation failed
    """
    try:
        # Create a larger figure to accommodate annotations without overlapping
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Convert PIL Image to numpy array for matplotlib
        img_array = np.array(image)
        
        # Get image dimensions
        img_height, img_width = img_array.shape[:2]
        
        # Create a larger canvas with padding around the image
        # This ensures annotations don't overlap with the image
        canvas_width = img_width * 1.5
        canvas_height = img_height * 1.5
        
        # Center the image on the canvas
        x_offset = (canvas_width - img_width) / 2
        y_offset = (canvas_height - img_height) / 2
        
        # Set the axis limits to create padding around the image
        ax.set_xlim(0, canvas_width)
        ax.set_ylim(canvas_height, 0)  # Inverted y-axis for image coordinates
        
        # Display the image centered on the canvas
        ax.imshow(img_array, extent=[x_offset, x_offset + img_width, y_offset + img_height, y_offset])
        ax.axis('off')  # Hide axes
        
        # Add title with product name
        product_title = product_details.get("title", "Product Features")
        plt.title(product_title, fontsize=14, fontweight='bold', pad=20)
        
        # Define annotation positions based on location strings
        # These are now positioned outside the image boundaries
        location_map = {
            "top-left": (x_offset * 0.5, y_offset * 0.5),
            "top": (x_offset + img_width/2, y_offset * 0.5),
            "top-right": (x_offset + img_width + x_offset * 0.5, y_offset * 0.5),
            "left": (x_offset * 0.5, y_offset + img_height/2),
            "center": (x_offset + img_width/2, y_offset + img_height/2),
            "right": (x_offset + img_width + x_offset * 0.5, y_offset + img_height/2),
            "bottom-left": (x_offset * 0.5, y_offset + img_height + y_offset * 0.5),
            "bottom": (x_offset + img_width/2, y_offset + img_height + y_offset * 0.5),
            "bottom-right": (x_offset + img_width + x_offset * 0.5, y_offset + img_height + y_offset * 0.5)
        }
        
        # Add annotations for each feature
        features = analysis.get("features", [])
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3']  # Different colors for annotations
        
        for i, feature in enumerate(features):
            # Get feature details
            name = feature.get("name", f"Feature {i+1}")
            description = feature.get("description", "")
            location_str = feature.get("location", "center").lower()
            confidence = feature.get("confidence", 90)
            
            # Find the closest predefined location
            closest_location = "center"
            for loc in location_map.keys():
                if loc in location_str:
                    closest_location = loc
                    break
            
            # Get position based on location string
            annotation_x, annotation_y = location_map.get(closest_location, location_map["center"])
            
            # Calculate a point on the image to point to (based on the location)
            if "top" in closest_location and "left" in closest_location:
                point_x, point_y = x_offset + img_width * 0.25, y_offset + img_height * 0.25
            elif "top" in closest_location and "right" in closest_location:
                point_x, point_y = x_offset + img_width * 0.75, y_offset + img_height * 0.25
            elif "bottom" in closest_location and "left" in closest_location:
                point_x, point_y = x_offset + img_width * 0.25, y_offset + img_height * 0.75
            elif "bottom" in closest_location and "right" in closest_location:
                point_x, point_y = x_offset + img_width * 0.75, y_offset + img_height * 0.75
            elif "top" in closest_location:
                point_x, point_y = x_offset + img_width * 0.5, y_offset + img_height * 0.25
            elif "bottom" in closest_location:
                point_x, point_y = x_offset + img_width * 0.5, y_offset + img_height * 0.75
            elif "left" in closest_location:
                point_x, point_y = x_offset + img_width * 0.25, y_offset + img_height * 0.5
            elif "right" in closest_location:
                point_x, point_y = x_offset + img_width * 0.75, y_offset + img_height * 0.5
            else:  # center
                point_x, point_y = x_offset + img_width * 0.5, y_offset + img_height * 0.5
            
            # Choose color for this annotation
            color = colors[i % len(colors)]
            
            # Add text annotation outside the image
            ax.text(
                annotation_x, annotation_y,
                f"{name}\n{description}",
                color='black',
                fontsize=12,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', edgecolor=color, alpha=0.9, boxstyle='round,pad=0.5', linewidth=2)
            )
            
            # Add arrow pointing to the feature
            ax.annotate(
                "",
                xy=(point_x, point_y),  # Point on the image
                xytext=(annotation_x, annotation_y),  # Text position
                arrowprops=dict(arrowstyle="->", color=color, lw=2, connectionstyle="arc3,rad=0.2"),
            )
        
        # Generate filename based on product title and image index
        safe_title = re.sub(r'[^\w\s-]', '', product_details.get("title", "product"))
        safe_title = re.sub(r'[-\s]+', '-', safe_title).strip('-').lower()
        filename = f"{safe_title}-annotated-{image_index}.jpg"
        filepath = os.path.join(output_path, filename)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated annotated image: {filepath}")
        return filepath
    
    except Exception as e:
        print(f"Error generating annotated image: {str(e)}")
        return None

def generate_review_scene(product_image: Image.Image, product_details: Dict[str, Any], output_path: str, image_index: int = 1) -> Optional[str]:
    """
    Generate a realistic review scene showing the product being used or reviewed
    
    Args:
        product_image (Image.Image): Original product image
        product_details (Dict[str, Any]): Product details
        output_path (str): Directory to save the generated image
        image_index (int): Index of the image being processed
        
    Returns:
        Optional[str]: Path to the generated image or None if generation failed
    """
    try:
        # Configure OpenAI client
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Convert image to base64
        buffered = BytesIO()
        product_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Get product information for context
        product_title = product_details.get("title", "Unknown Product")
        product_brand = product_details.get("brand", "Unknown Brand")
        product_category = product_details.get("categories", [])[0] if product_details.get("categories") else "product"
        
        # Determine appropriate background based on product category
        background_setting = "studio setup with neutral background"
        
        # Customize background based on product type
        if any(keyword in product_title.lower() for keyword in ["mouse", "keyboard", "headset", "gaming"]):
            background_setting = "professional gaming setup with RGB lighting and gaming monitors"
        elif any(keyword in product_title.lower() for keyword in ["phone", "smartphone", "mobile"]):
            background_setting = "modern desk with minimal design"
        elif any(keyword in product_title.lower() for keyword in ["laptop", "computer", "pc"]):
            background_setting = "clean workspace with desk and computer peripherals"
        elif any(keyword in product_title.lower() for keyword in ["camera", "lens", "photo"]):
            background_setting = "photography studio with lighting equipment"
        elif any(keyword in product_title.lower() for keyword in ["watch", "jewelry"]):
            background_setting = "elegant display with soft lighting"
        elif any(keyword in product_title.lower() for keyword in ["shoe", "sneaker", "footwear"]):
            background_setting = "shoe display with wooden floor"
        elif any(keyword in product_title.lower() for keyword in ["kitchen", "cookware", "appliance"]):
            background_setting = "modern kitchen countertop"
        
        # Create prompt for image generation
        prompt = f"""Create a photorealistic product showcase image of this {product_category}: {product_title} by {product_brand}. 
        
Show the product from a different angle in a {background_setting}. DO NOT include any human faces or people in the image. 
        
The scene should look like a professional product photography setup with excellent lighting to highlight the product features. Make sure the product is clearly visible and is the main focus of the image.
        
The image should be clean, minimalist, and suitable for a professional product review."""
        
        print("Generating professional product showcase with DALL-E...")
        
        # Generate image with DALL-E
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        
        # Get the image URL from the response
        image_url = response.data[0].url
        
        # Download the generated image
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            # Save the image
            safe_title = re.sub(r'[^\w\s-]', '', product_details.get("title", "product"))
            safe_title = re.sub(r'[-\s]+', '-', safe_title).strip('-').lower()
            filename = f"{safe_title}-product-showcase-{image_index}.jpg"
            filepath = os.path.join(output_path, filename)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            print(f"Generated professional product showcase: {filepath}")
            return filepath
        else:
            print(f"Failed to download generated image: HTTP {response.status_code}")
            return None
    
    except Exception as e:
        print(f"Error generating product showcase: {str(e)}")
        return None

def generate_product_video(product_details: Dict[str, Any], review: Dict[str, Any], 
                          duration: int = 60, with_voiceover: bool = False,
                          output_dir: str = "output") -> Optional[str]:
    """
    Generate a video review of the product
    
    Args:
        product_details (Dict[str, Any]): Product details dictionary
        review (Dict[str, Any]): Generated review content
        duration (int): Duration of the video in seconds
        with_voiceover (bool): Whether to add AI-generated voiceover
        output_dir (str): Directory to save the generated video
        
    Returns:
        Optional[str]: Path to the generated video or None if generation failed
    """
    if not OPENAI_API_KEY:
        print("Error: OpenAI API key is required for video generation. Set the OPENAI_API_KEY environment variable.")
        return None
        
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a script for the video based on the review
        print("Generating video script...")
        script = generate_video_script(review, duration)
        
        # If voiceover is requested, generate audio narration
        audio_path = None
        if with_voiceover:
            print("Generating voiceover...")
            audio_path = generate_voiceover(script, output_dir)
        
        # Collect images for the video
        print("Collecting images for video...")
        image_paths = []
        
        # Use product images if available
        if product_details.get("image_urls"):
            # First check if we already have processed images
            existing_images = [f for f in os.listdir(output_dir) if f.startswith("product_") and f.endswith(".png")]
            
            if existing_images:
                image_paths = [os.path.join(output_dir, img) for img in existing_images]
            else:
                # Process images if we haven't already
                image_paths = process_product_images(product_details, output_dir)
        
        # If we don't have enough images, generate some additional ones
        if len(image_paths) < 3:
            print("Generating additional product images...")
            for i in range(3 - len(image_paths)):
                # Download the first product image if available
                if product_details.get("image_urls") and len(product_details["image_urls"]) > 0:
                    base_image = download_image(product_details["image_urls"][0])
                    if base_image:
                        # Generate a review scene with the product
                        scene_path = generate_review_scene(base_image, product_details, output_dir, i+1)
                        if scene_path:
                            image_paths.append(scene_path)
        
        # Create the video from images and audio
        print(f"Creating {duration} second video...")
        video_path = create_video_from_images(image_paths, script, audio_path, duration, output_dir)
        
        return video_path
    
    except Exception as e:
        print(f"Error generating video: {str(e)}")
        return None

def generate_video_script(review: Dict[str, Any], duration: int = 60) -> str:
    """
    Generate a script for the video based on the review
    
    Args:
        review (Dict[str, Any]): Generated review content
        duration (int): Target duration of the video in seconds
        
    Returns:
        str: Script for the video
    """
    content = review["content"]
    product_title = review["product_title"]
    
    # Calculate approximate word count based on speaking rate (150 words per minute)
    target_word_count = (duration / 60) * 150
    
    # Create a prompt for OpenAI to generate a script
    prompt = f"""Create a concise {duration}-second video script for a product review of '{product_title}'.
    The script should be approximately {int(target_word_count)} words and cover these key points:
    
    Introduction: {content.get('introduction', 'Brief introduction to the product')}
    
    Pros: {content.get('pros', 'Main advantages of the product')}
    
    Cons: {content.get('cons', 'Main disadvantages of the product')}
    
    Verdict: {content.get('verdict', 'Final verdict on the product')}
    
    Format the script as a continuous paragraph that flows naturally when spoken.
    """
    
    try:
        # Call OpenAI API to generate the script
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional product reviewer who creates engaging video scripts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        script = response.choices[0].message.content.strip()
        return script
    
    except Exception as e:
        print(f"Error generating video script: {str(e)}")
        # Fallback to a simple script based on the review content
        script_parts = []
        
        if content.get("introduction"):
            script_parts.append(content["introduction"])
        
        if content.get("pros"):
            script_parts.append("The main advantages of this product are: " + content["pros"])
        
        if content.get("cons"):
            script_parts.append("However, there are some drawbacks: " + content["cons"])
        
        if content.get("verdict"):
            script_parts.append("In conclusion: " + content["verdict"])
        
        return " ".join(script_parts)

def generate_voiceover(script: str, output_dir: str) -> Optional[str]:
    """
    Generate an AI voiceover for the video script
    
    Args:
        script (str): The video script
        output_dir (str): Directory to save the audio file
        
    Returns:
        Optional[str]: Path to the generated audio file or None if generation failed
    """
    try:
        # For now, print a message that this would use an API like ElevenLabs or similar
        print("Note: This would use a Text-to-Speech API to generate a voiceover")
        print(f"Script for voiceover ({len(script.split())} words):\n{script[:100]}...")
        
        # Create a placeholder audio file path
        audio_path = os.path.join(output_dir, "voiceover.mp3")
        
        # In a real implementation, you would call a TTS API here
        # For now, we'll just return the path as if it was created
        
        return audio_path
    
    except Exception as e:
        print(f"Error generating voiceover: {str(e)}")
        return None

def create_video_from_images(image_paths: List[str], script: str, audio_path: Optional[str], 
                            duration: int, output_dir: str) -> Optional[str]:
    """
    Create a video from a collection of images and optional audio
    
    Args:
        image_paths (List[str]): Paths to the images to include in the video
        script (str): The video script (used for captions if no audio)
        audio_path (Optional[str]): Path to the voiceover audio file
        duration (int): Duration of the video in seconds
        output_dir (str): Directory to save the video
        
    Returns:
        Optional[str]: Path to the generated video or None if generation failed
    """
    try:
        # For now, print a message that this would use a library like MoviePy
        print("Note: This would use MoviePy or a similar library to create the video")
        print(f"Creating video with {len(image_paths)} images over {duration} seconds")
        
        # Create a placeholder video file path
        video_filename = f"product_review_{int(time.time())}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        # In a real implementation, you would use MoviePy or similar to create the video
        # For now, we'll just return the path as if it was created
        
        # Create an empty file as a placeholder
        with open(video_path, 'w') as f:
            f.write(f"This is a placeholder for a {duration}-second product review video")
            f.write(f"\nIt would include {len(image_paths)} images and")
            f.write(f"\n{'a voiceover' if audio_path else 'captions'}")
        
        return video_path
    
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return None

def format_output(review: Dict[str, Any], format_json: bool = False) -> str:
    """
    Format the review output for terminal display
    
    Args:
        review (dict): Review data
        format_json (bool): Whether to output as JSON
        
    Returns:
        str: Formatted output string
    """
    if format_json:
        return json.dumps(review, indent=2, ensure_ascii=False)
    
    if "error" in review:
        return f"\nERROR: {review['error']}"
    
    output = []
    output.append("\n" + "="*80)
    output.append(f"AUTOMATED REVIEW: {review['product_title']}")
    output.append(f"Type: {review['review_type'].capitalize()} Review")
    output.append(f"Generated: {review['generated_at']}")
    output.append("="*80)
    
    content = review["content"]
    
    # Introduction
    if "introduction" in content:
        output.append("\nINTRODUCTION:")
        output.append("-"*80)
        output.append(content["introduction"])
    
    # Pros
    if "pros" in content:
        output.append("\nPROS:")
        output.append("-"*80)
        output.append(content["pros"])
    
    # Cons
    if "cons" in content:
        output.append("\nCONS:")
        output.append("-"*80)
        output.append(content["cons"])
    
    # Verdict
    if "verdict" in content:
        output.append("\nVERDICT:")
        output.append("-"*80)
        output.append(content["verdict"])
    
    # Rating
    if "rating" in content:
        output.append(f"\nRATING: {content['rating']}/10")
    
    # Full text (for debugging)
    # output.append("\nFULL TEXT:")
    # output.append("-"*80)
    # output.append(content["full_text"])
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description='Extract product details and generate automated reviews from Amazon URLs')
    parser.add_argument('url', help='Amazon product URL')
    parser.add_argument('--output', '-o', help='Output file path (JSON format)')
    parser.add_argument('--json', '-j', action='store_true', help='Display output as JSON')
    parser.add_argument('--review-type', '-t', choices=['detailed', 'summary', 'comparison'], 
                        default='detailed', help='Type of review to generate')
    parser.add_argument('--generate-image', '-i', action='store_true', 
                        help='Generate annotated product image with feature highlights')
    parser.add_argument('--image-output-dir', '-d', default='output',
                        help='Directory to save generated images (default: "output")')
    parser.add_argument('--generate-video', '-v', action='store_true',
                        help='Generate video review of the product')
    parser.add_argument('--video-duration', type=int, default=60,
                        help='Duration of the generated video in seconds (default: 60)')
    parser.add_argument('--with-voiceover', action='store_true',
                        help='Add AI-generated voiceover to the video')
    args = parser.parse_args()
    
    # Validate URL - check for both amazon.com and shortened amzn links
    if not ('amazon.' in args.url or 'amzn.' in args.url):
        print("Error: The URL doesn't appear to be from Amazon.")
        sys.exit(1)
    
    # Fetch product details
    product_details = fetch_amazon_product_details(args.url)
    
    if "error" in product_details:
        print(f"Error: {product_details['error']}")
        sys.exit(1)
    
    # Generate review
    print(f"Generating {args.review_type} review...")
    review = generate_product_review(product_details, args.review_type)
    
    # Display results
    print(format_output(review, args.json))
    
    # Save to file if specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(review, f, indent=2, ensure_ascii=False)
            print(f"\nReview saved to: {args.output}")
        except Exception as e:
            print(f"\nError saving to file: {str(e)}")
    
    # Process product images if requested
    if args.generate_image and product_details.get("image_urls"):
        print("\nProcessing product images...")
        image_paths = process_product_images(product_details, args.image_output_dir)
        if image_paths:
            print(f"Generated images saved to: {args.image_output_dir}")
    
    # Generate video if requested
    if args.generate_video:
        print("\nGenerating product review video...")
        video_path = generate_product_video(product_details, review, 
                                          duration=args.video_duration,
                                          with_voiceover=args.with_voiceover,
                                          output_dir=args.image_output_dir)
        if video_path:
            print(f"Generated video saved to: {video_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)
