import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

# Import the humanizer functions
from humanizer import humanize_text as humanize_text_function
from humanizer import echo_text as echo_text_function

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("text-humanizer-api")

# Create FastAPI app
app = FastAPI(
    title="Text Humanizer API",
    description="API service for humanizing text using OpenAI's fine-tuned models",
    version="1.0.0"
)

# Define request models
class TextRequest(BaseModel):
    input_text: str

@app.get("/")
async def read_root():
    return {"message": "Text Humanizer API is running. Use /humanize_text or /echo_text endpoints."}

@app.post("/humanize_text")
async def humanize_text(request: TextRequest):
    try:
        logger.info(f"Received request to humanize text: {request.input_text[:50]}...")
        result = await humanize_text_function(request.input_text)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in humanize_text endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/echo_text")
async def echo_text(request: TextRequest):
    try:
        logger.info(f"Received request to echo text: {request.input_text[:50]}...")
        result = await echo_text_function(request.input_text)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in echo_text endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"}
    )

# Run the server if executed as a script
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
