#!/usr/bin/env python3
"""
Amazon Product Scraper API
------------------------
A FastAPI API to extract detailed information from Amazon product URLs.

Usage:
    uvicorn api:app --reload
    or
    python api.py

Endpoints:
    GET /api/product?url=<amazon_product_url>
    POST /api/product with JSON body {"url": "<amazon_product_url>"}
"""

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Union, Dict, Any, List
from scrape import fetch_amazon_product_details
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Define request model
class ProductRequest(BaseModel):
    url: str = Field(..., description="Amazon product URL")

# Define response models
class ProductResponse(BaseModel):
    title: Optional[str] = None
    price: Optional[str] = None
    currency: Optional[str] = None
    description: Optional[str] = None
    features: List[str] = []
    specifications: Dict[str, str] = {}
    rating: Optional[Union[str, float]] = None
    review_count: Optional[str] = None
    image_urls: List[str] = []
    availability: Optional[str] = None
    seller: Optional[str] = None
    brand: Optional[str] = None
    categories: List[str] = []
    url: str

class ErrorResponse(BaseModel):
    error: str
    message: str

# Create FastAPI app
app = FastAPI(
    title="Amazon Product Scraper API",
    description="API to extract detailed information from Amazon product URLs",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/product", response_model=Union[ProductResponse, ErrorResponse], tags=["Product"])
async def get_product_details(url: str = Query(..., description="Amazon product URL")):
    """
    Get product details from an Amazon URL using GET request
    """
    return await process_product_url(url)

@app.post("/api/product", response_model=Union[ProductResponse, ErrorResponse], tags=["Product"])
async def post_product_details(request: ProductRequest):
    """
    Get product details from an Amazon URL using POST request
    """
    return await process_product_url(request.url)

async def process_product_url(url: str):
    """
    Process the Amazon product URL and return the product details
    """
    # Validate URL (basic check)
    if not ('amazon.' in url or 'amzn.' in url):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid URL",
                "message": "The URL doesn't appear to be from Amazon"
            }
        )
    
    try:
        # Fetch product details
        product_details = fetch_amazon_product_details(url)
        
        # Check if there was an error
        if isinstance(product_details, dict) and "error" in product_details:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Scraping error",
                    "message": product_details["error"]
                }
            )
        
        # Remove top_reviews from the response as per user's request
        if isinstance(product_details, dict) and "top_reviews" in product_details:
            del product_details["top_reviews"]
        
        return product_details
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Server error",
                "message": str(e)
            }
        )

@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse, tags=["Documentation"])
async def index():
    """
    API documentation page
    """
    return """
    <html>
        <head>
            <title>Amazon Product Scraper API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
                h1, h2 { color: #333; }
                .endpoint { margin-bottom: 20px; }
                .note { background-color: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Amazon Product Scraper API</h1>
            <p>This API allows you to extract detailed information from Amazon product URLs.</p>
            
            <div class="note">
                <p><strong>Note:</strong> For interactive API documentation, visit <a href="/docs">/docs</a> or <a href="/redoc">/redoc</a></p>
            </div>
            
            <h2>Endpoints</h2>
            
            <div class="endpoint">
                <h3>GET /api/product</h3>
                <p>Fetch product details using a URL query parameter.</p>
                <pre>GET /api/product?url=https://www.amazon.com/dp/B07PXGQC1Q</pre>
            </div>
            
            <div class="endpoint">
                <h3>POST /api/product</h3>
                <p>Fetch product details by sending a JSON body.</p>
                <pre>POST /api/product
Content-Type: application/json

{
    "url": "https://www.amazon.com/dp/B07PXGQC1Q"
}</pre>
            </div>
            
            <div class="endpoint">
                <h3>GET /api/health</h3>
                <p>Health check endpoint.</p>
                <pre>GET /api/health</pre>
            </div>
        </body>
    </html>
    """

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=os.environ.get("DEBUG", "False").lower() == "true")
