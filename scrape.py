#!/usr/bin/env python3
"""
Amazon Product Scraper
---------------------
A standalone script to extract detailed information from Amazon product URLs.

Usage:
    python amazon_scraper.py <amazon_product_url> [--output filename.json]

Example:
    python amazon_scraper.py https://www.amazon.com/dp/B07PXGQC1Q --output product_data.json
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

def fetch_amazon_product_details(url):
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
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
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
            # Try alternative description location
            description_element = soup.select_one('#feature-bullets')
            if description_element:
                product_details["description"] = description_element.text.strip()
        
        # Extract product features
        feature_elements = soup.select('#feature-bullets .a-list-item')
        for feature in feature_elements:
            feature_text = feature.text.strip()
            if feature_text and not feature_text.startswith("›"):
                product_details["features"].append(feature_text)
        
        # Extract product specifications
        spec_tables = soup.select('table.a-normal, table.a-keyvalue')
        for table in spec_tables:
            rows = table.select('tr')
            for row in rows:
                key_element = row.select_one('th, td.a-span3, td:first-child')
                value_element = row.select_one('td.a-span9, td:last-child')
                if key_element and value_element:
                    key = key_element.text.strip().rstrip(':')
                    value = value_element.text.strip()
                    if key and value:
                        product_details["specifications"][key] = value
        
        # Also check the product details section
        detail_section = soup.select_one('#detailBullets_feature_div')
        if detail_section:
            list_items = detail_section.select('li')
            for item in list_items:
                spans = item.select('span')
                if len(spans) >= 2:
                    key = spans[0].text.strip().replace(':', '').strip()
                    value = spans[1].text.strip()
                    if key and value:
                        product_details["specifications"][key] = value
        
        # Extract additional specifications from the productDetails section
        spec_section = soup.select_one('#productDetails_techSpec_section_1')
        if spec_section:
            rows = spec_section.select('tr')
            for row in rows:
                key = row.select_one('th')
                value = row.select_one('td')
                if key and value:
                    product_details["specifications"][key.text.strip()] = value.text.strip()
        
        # Extract brand
        brand_element = soup.select_one('#bylineInfo')
        if brand_element:
            brand_text = brand_element.text.strip()
            brand_match = re.search(r'Brand:\s*(\w+)', brand_text)
            if brand_match:
                product_details["brand"] = brand_match.group(1)
            else:
                product_details["brand"] = brand_text.replace('Visit the ', '').replace(' Store', '')
        
        # Extract availability
        availability_element = soup.select_one('#availability')
        if availability_element:
            product_details["availability"] = availability_element.text.strip()
        
        # Extract seller info
        seller_element = soup.select_one('#merchant-info')
        if seller_element:
            product_details["seller"] = seller_element.text.strip()
        
        # Extract rating
        rating_element = soup.select_one('#acrPopover')
        if rating_element and 'title' in rating_element.attrs:
            rating_text = rating_element['title']
            rating_match = re.search(r'(\d+\.?\d*) out of \d+ stars', rating_text)
            if rating_match:
                product_details["rating"] = float(rating_match.group(1))
        
        # Extract review count
        review_count_element = soup.select_one('#acrCustomerReviewText')
        if review_count_element:
            review_text = review_count_element.text.strip()
            count_match = re.search(r'(\d+(?:,\d+)*)', review_text)
            if count_match:
                product_details["review_count"] = count_match.group(1).replace(',', '')
        
        # Extract top reviews
        review_elements = soup.select('#cm-cr-dp-review-list .review')
        for review in review_elements[:5]:  # Get up to 5 top reviews
            review_rating_element = review.select_one('.review-rating')
            review_title_element = review.select_one('.review-title')
            review_text_element = review.select_one('.review-text')
            review_author_element = review.select_one('.a-profile-name')
            review_date_element = review.select_one('.review-date')
            
            review_data = {"rating": None, "title": None, "text": None, "author": None, "date": None}
            
            if review_rating_element:
                rating_text = review_rating_element.text.strip()
                rating_match = re.search(r'(\d+\.?\d*) out of \d+ stars', rating_text)
                if rating_match:
                    review_data["rating"] = float(rating_match.group(1))
                else:
                    review_data["rating"] = rating_text
            
            if review_title_element:
                review_data["title"] = review_title_element.text.strip()
            
            if review_text_element:
                review_data["text"] = review_text_element.text.strip()
            
            if review_author_element:
                review_data["author"] = review_author_element.text.strip()
            
            if review_date_element:
                review_data["date"] = review_date_element.text.strip()
            
            if any(review_data.values()):  # Only add if any data was found
                product_details["top_reviews"].append(review_data)
        
        # Extract product categories
        category_elements = soup.select('#wayfinding-breadcrumbs_feature_div li')
        for category in category_elements:
            category_text = category.text.strip()
            if category_text and category_text != '›':
                product_details["categories"].append(category_text)
        
        # Extract product images
        # First try to find the main image
        main_image = soup.select_one('#landingImage')
        if main_image and 'data-old-hires' in main_image.attrs:
            product_details["image_urls"].append(main_image['data-old-hires'])
        elif main_image and 'src' in main_image.attrs:
            product_details["image_urls"].append(main_image['src'])
        
        # Then try to extract more images from scripts
        image_pattern = re.compile(r'"hiRes":"([^"]+)"')
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string and 'ImageBlockATF' in str(script.string):
                matches = image_pattern.findall(str(script.string))
                for match in matches:
                    if match not in product_details["image_urls"]:
                        product_details["image_urls"].append(match)
        
        print(f"Successfully extracted product data for: {product_details['title']}")
        return product_details
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Request error: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def format_output(product_details, format_json=False):
    """
    Format the output for terminal display
    """
    if format_json:
        return json.dumps(product_details, indent=2, ensure_ascii=False)
    
    if "error" in product_details:
        return f"\nERROR: {product_details['error']}"
    
    output = []
    output.append("\n" + "="*80)
    output.append(f"PRODUCT: {product_details['title']}")
    output.append("="*80)
    
    # Price and basic info
    price_str = f"{product_details['currency'] or '$'}{product_details['price']}" if product_details['price'] else "Price not available"
    output.append(f"\nPrice: {price_str}")
    
    if product_details['brand']:
        output.append(f"Brand: {product_details['brand']}")
    
    if product_details['availability']:
        output.append(f"Availability: {product_details['availability']}")
    
    if product_details['rating']:
        stars = "★" * int(product_details['rating']) + "☆" * (5 - int(product_details['rating']))
        output.append(f"Rating: {stars} ({product_details['rating']}/5 from {product_details['review_count'] or 'unknown'} reviews)")
    
    # Categories
    if product_details['categories']:
        output.append("\nCategories:")
        output.append(" > ".join(product_details['categories']))
    
    # Description
    if product_details['description']:
        output.append("\nDESCRIPTION:")
        output.append("-"*80)
        output.append(product_details['description'])
    
    # Features
    if product_details['features']:
        output.append("\nFEATURES:")
        output.append("-"*80)
        for i, feature in enumerate(product_details['features'], 1):
            output.append(f"{i}. {feature}")
    
    # Specifications
    if product_details['specifications']:
        output.append("\nSPECIFICATIONS:")
        output.append("-"*80)
        for key, value in product_details['specifications'].items():
            output.append(f"• {key}: {value}")
    
    # Reviews
    if product_details['top_reviews']:
        output.append("\nTOP REVIEWS:")
        output.append("-"*80)
        for i, review in enumerate(product_details['top_reviews'], 1):
            output.append(f"Review #{i}:")
            if review['rating']:
                output.append(f"Rating: {review['rating']}")
            if review['title']:
                output.append(f"Title: {review['title']}")
            if review['author']:
                output.append(f"By: {review['author']}")
            if review['date']:
                output.append(f"Date: {review['date']}")
            if review['text']:
                output.append(f"Comment: {review['text']}")
            output.append("")
    
    # Images
    if product_details['image_urls']:
        output.append("\nIMAGE URLS:")
        output.append("-"*80)
        for i, url in enumerate(product_details['image_urls'], 1):
            output.append(f"Image #{i}: {url}")
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description='Extract product details from Amazon URLs')
    parser.add_argument('url', help='Amazon product URL')
    parser.add_argument('--output', '-o', help='Output file path (JSON format)')
    parser.add_argument('--json', '-j', action='store_true', help='Display output as JSON')
    args = parser.parse_args()
    
    # Validate URL
    if not ('amazon.' in args.url or 'amzn.' in args.url):
        print("Error: The URL doesn't appear to be from Amazon.")
        sys.exit(1)
    
    # Fetch product details
    product_details = fetch_amazon_product_details(args.url)
    
    # Display results
    print(format_output(product_details, args.json))
    
    # Save to file if specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(product_details, f, indent=2, ensure_ascii=False)
            print(f"\nProduct details saved to: {args.output}")
        except Exception as e:
            print(f"\nError saving to file: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)