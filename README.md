# Amazon Product Review Generator API

This API allows you to extract detailed information from Amazon product URLs. It provides a simple HTTP interface to the product scraper functionality using FastAPI.

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the API Server

You can start the API server using Uvicorn:

```bash
# Using Uvicorn directly
uvicorn api:app --reload

# OR using Python script
python api.py
```

By default, the server will run on port 8000. You can change this by setting the `PORT` environment variable.

### API Endpoints

#### GET /api/product

Fetch product details using a URL query parameter.

```
GET /api/product?url=https://www.amazon.com/dp/B07PXGQC1Q
```

#### POST /api/product

Fetch product details by sending a JSON body.

```
POST /api/product
Content-Type: application/json

{
    "url": "https://www.amazon.com/dp/B07PXGQC1Q"
}
```

#### GET /api/health

Health check endpoint.

```
GET /api/health
```

#### GET /

API documentation page (HTML).

#### GET /docs

Interactive API documentation (Swagger UI).

#### GET /redoc

Alternative API documentation (ReDoc).

## Environment Variables

- `PORT`: Port number for the API server (default: 8000)
- `DEBUG`: Set to "true" to enable debug mode (default: "False")

## Response Format

The API returns JSON responses with the following structure for successful requests:

```json
{
  "title": "Product Title",
  "price": "19.99",
  "currency": "$",
  "description": "Product description...",
  "features": ["Feature 1", "Feature 2"],
  "specifications": {"Spec1": "Value1", "Spec2": "Value2"},
  "rating": "4.5",
  "review_count": "1234",
  "top_reviews": [
    {
      "rating": "5",
      "title": "Great product",
      "author": "John Doe",
      "date": "January 1, 2023",
      "text": "Review text..."
    }
  ],
  "image_urls": ["https://example.com/image1.jpg"],
  "availability": "In Stock",
  "seller": "Amazon",
  "brand": "Brand Name",
  "categories": ["Category 1", "Category 2"],
  "url": "https://www.amazon.com/dp/B07PXGQC1Q"
}
```

For error responses, the API returns:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

## Error Handling

The API handles various error scenarios:

- Missing URL parameter (400 Bad Request)
- Invalid Amazon URL (400 Bad Request)
- Scraping errors (500 Internal Server Error)
- Server errors (500 Internal Server Error)

## Advantages of FastAPI

- **Performance**: FastAPI is built on Starlette and Pydantic, offering high performance.
- **Type Hints**: Uses Python type hints for request/response validation.
- **Automatic Documentation**: Generates interactive API documentation (Swagger UI and ReDoc).
- **Modern Python**: Takes advantage of Python 3.6+ features.
- **Asynchronous Support**: Built-in support for async/await syntax.
