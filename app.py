import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Text Humanizer API</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                pre {
                    background: #f8f8f8;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    padding: 15px;
                    overflow-x: auto;
                }
                .form-group {
                    margin-bottom: 15px;
                }
                textarea {
                    width: 100%;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-family: inherit;
                    height: 100px;
                }
                button {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 15px;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #2980b9;
                }
                .result {
                    margin-top: 20px;
                    padding: 15px;
                    background: #f1f9ff;
                    border-radius: 4px;
                    display: none;
                }
            </style>
        </head>
        <body>
            <h1>Text Humanizer API</h1>
            <p>An API service for humanizing text using OpenAI's fine-tuned models.</p>
            
            <h2>Try it out:</h2>
            <div class="form-group">
                <textarea id="inputText" placeholder="Enter your text here..."></textarea>
            </div>
            <div class="form-group">
                <button onclick="humanizeText()">Humanize Text</button>
                <button onclick="echoText()">Echo Text</button>
            </div>
            
            <div id="result" class="result">
                <h3>Result:</h3>
                <p id="resultText"></p>
            </div>
            
            <h2>API Documentation</h2>
            <p>Check out the <a href="/docs">API documentation</a> for more details on how to use the API.</p>
            
            <h2>API Endpoints</h2>
            <h3>POST /humanize_text</h3>
            <p>Transforms input text into more natural, human-like writing.</p>
            <pre>
curl -X POST "https://web-production-3db6c.up.railway.app/humanize_text" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Your text to humanize"}'
            </pre>
            
            <h3>POST /echo_text</h3>
            <p>Echoes back the input text (useful for testing).</p>
            <pre>
curl -X POST "https://web-production-3db6c.up.railway.app/echo_text" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Text to echo"}'
            </pre>

            <script>
                async function humanizeText() {
                    const inputText = document.getElementById('inputText').value;
                    if (!inputText) {
                        alert('Please enter some text');
                        return;
                    }
                    
                    try {
                        const response = await fetch('/humanize_text', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ input_text: inputText })
                        });
                        
                        const data = await response.json();
                        document.getElementById('resultText').textContent = data.result;
                        document.getElementById('result').style.display = 'block';
                    } catch (error) {
                        alert('Error: ' + error.message);
                    }
                }
                
                async function echoText() {
                    const inputText = document.getElementById('inputText').value;
                    if (!inputText) {
                        alert('Please enter some text');
                        return;
                    }
                    
                    try {
                        const response = await fetch('/echo_text', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ input_text: inputText })
                        });
                        
                        const data = await response.json();
                        document.getElementById('resultText').textContent = data.result;
                        document.getElementById('result').style.display = 'block';
                    } catch (error) {
                        alert('Error: ' + error.message);
                    }
                }
            </script>
        </body>
    </html>
    """
    return html_content

@app.get("/echo_text")
async def echo_text_get(text: str = Query(None, description="Text to echo")):
    if not text:
        return {"message": "Please provide text using the ?text=your_text query parameter"}
    
    try:
        logger.info(f"Received GET request to echo text: {text[:50]}...")
        result = await echo_text_function(text)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in echo_text endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/echo_text")
async def echo_text_post(request: TextRequest):
    try:
        logger.info(f"Received POST request to echo text: {request.input_text[:50]}...")
        result = await echo_text_function(request.input_text)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in echo_text endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/humanize_text")
async def humanize_text_get(text: str = Query(None, description="Text to humanize")):
    if not text:
        return {"message": "Please provide text using the ?text=your_text query parameter"}
    
    try:
        logger.info(f"Received GET request to humanize text: {text[:50]}...")
        result = await humanize_text_function(text)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in humanize_text endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/humanize_text")
async def humanize_text_post(request: TextRequest):
    try:
        logger.info(f"Received POST request to humanize text: {request.input_text[:50]}...")
        result = await humanize_text_function(request.input_text)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in humanize_text endpoint: {str(e)}")
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
