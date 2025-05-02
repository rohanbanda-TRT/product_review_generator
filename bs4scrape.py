import requests
from bs4 import BeautifulSoup
import re

def get_amazon_product_details(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.93 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {"error": f"Failed to fetch page. Status code: {response.status_code}"}

    soup = BeautifulSoup(response.content, "html.parser")

    def extract_text(selector):
        element = soup.select_one(selector)
        return element.get_text(strip=True) if element else None

    title = extract_text("#productTitle")
    price = extract_text(".a-price .a-offscreen")
    rating = extract_text("span.a-icon-alt")
    reviews = extract_text("#acrCustomerReviewText")
    availability = extract_text("#availability .a-declarative, #availability span")
    brand = extract_text("#bylineInfo")
    
    # Product details table
    details = {}
    for row in soup.select("#productDetails_techSpec_section_1 tr, #productDetails_detailBullets_sections1 tr"):
        heading = row.select_one("th, td")
        value = row.select("td")
        if heading and value:
            details[heading.get_text(strip=True)] = value[-1].get_text(strip=True)

    # Description
    description = extract_text("#productDescription p") or extract_text("#productDescription")

    # Images (using regex to get from imageBlockData)
    image_urls = []
    image_data_script = soup.find("script", text=re.compile("ImageBlockATF"))
    if image_data_script:
        image_matches = re.findall(r'"hiRes":"(https[^"]+)"', image_data_script.string)
        image_urls = list(set(image_matches))  # remove duplicates

    return {
        "title": title,
        "price": price,
        "rating": rating,
        "number_of_reviews": reviews,
        "availability": availability,
        "brand": brand,
        "product_description": description,
        "product_details": details,
        "images": image_urls,
    }

# Example usage
url = "https://amzn.in/d/4hXpOnb"  # Replace with actual product URL
product_data = get_amazon_product_details(url)
print(product_data)
