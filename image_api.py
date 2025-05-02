from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import uvicorn
import os
import uuid
from image_generator import ImageGenerator
import shutil

# Create FastAPI app
app = FastAPI(title="Image Generation API", description="API for generating images using OpenAI's Vision model")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize image generator
generator = ImageGenerator()

# Create directory for uploaded images if it doesn't exist
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Generation API"}

@app.post("/generate-from-prompt")
async def generate_from_prompt(
    prompt: str = Form(...),
    model: str = Form("dall-e-3"),
    size: str = Form("1024x1024"),
    quality: str = Form("standard"),
    n: int = Form(1)
):
    """Generate image based on text prompt only"""
    try:
        image_urls = generator.generate_image(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            n=n
        )
        
        if not image_urls:
            raise HTTPException(status_code=500, detail="Failed to generate images")
        
        return JSONResponse(content={"urls": image_urls})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-from-image")
async def generate_from_image(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    image: UploadFile = File(...),
    model: str = Form("dall-e-3"),
    size: str = Form("1024x1024"),
    quality: str = Form("standard"),
    n: int = Form(1)
):
    """Generate image based on text prompt and uploaded image"""
    try:
        # Create a unique filename
        file_extension = os.path.splitext(image.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Generate image
        image_urls = generator.generate_image(
            prompt=prompt,
            image_path=file_path,
            model=model,
            size=size,
            quality=quality,
            n=n
        )
        
        # Schedule file cleanup
        background_tasks.add_task(os.remove, file_path)
        
        if not image_urls:
            raise HTTPException(status_code=500, detail="Failed to generate images")
        
        return JSONResponse(content={"urls": image_urls})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-from-image-url")
async def generate_from_image_url(
    prompt: str = Form(...),
    image_url: str = Form(...),
    model: str = Form("dall-e-3"),
    size: str = Form("1024x1024"),
    quality: str = Form("standard"),
    n: int = Form(1)
):
    """Generate image based on text prompt and image URL"""
    try:
        # Generate image
        image_urls = generator.generate_image(
            prompt=prompt,
            image_path=image_url,
            model=model,
            size=size,
            quality=quality,
            n=n
        )
        
        if not image_urls:
            raise HTTPException(status_code=500, detail="Failed to generate images")
        
        return JSONResponse(content={"urls": image_urls})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("image_api:app", host="0.0.0.0", port=8000, reload=True)
