# Text Humanizer API

An API service for humanizing text using OpenAI's fine-tuned models.

## Overview

This service provides an API endpoint to transform text into more natural, human-like writing. It uses OpenAI's fine-tuned models to rewrite sentences while preserving the original meaning.

## Features

- Text segmentation into sentences with special case handling
- Multiple fine-tuned models for different rewriting styles
- Best sentence selection using GPT-4o-mini
- Final cleaning pass with GPT-4o
- Rate limiting and API key rotation
- Async processing for better performance

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key(s)

### Installation

1. Clone the repository
   ```
   git clone https://github.com/granitevolition/text-humanizer-api.git
   cd text-humanizer-api
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key(s)
   ```
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

### Running Locally

```
python humanizer.py
```

The server will start on port 8000 by default (configurable in `.env`).

## API Endpoints

### `/humanize_text`

Transforms input text into more natural, human-like writing.

- **Method**: POST
- **Parameters**: 
  - `input_text` (string): The text to be humanized
- **Returns**: Humanized version of the input text

### `/echo_text`

Echoes back the input text (for testing).

- **Method**: POST
- **Parameters**: 
  - `input_text` (string): The text to echo
- **Returns**: The same input text

## Example Usage

Using cURL:

```bash
curl -X POST http://localhost:8000/humanize_text \
  -H "Content-Type: application/json" \
  -d '{"input_text": "The cat sat on the mat. It was very comfortable."}'
```

Using Python with requests:

```python
import requests

response = requests.post(
    "http://localhost:8000/humanize_text",
    json={"input_text": "The cat sat on the mat. It was very comfortable."}
)

print(response.json())
```

## Configuration

The application can be configured using environment variables. See `.env.example` for all available options.

Key configurations:

- `OPENAI_API_KEY`: Your OpenAI API key
- `DEFAULT_MODEL`: The fine-tuned model to use (default: ft:gpt-3.5-turbo-0125:personal::9hpCfvVt)
- `SELECTION_MODEL`: Model used to select the best sentence (default: gpt-4o-mini)
- `CLEANING_MODEL`: Model used for the final cleaning pass (default: gpt-4o)
- `TEMPERATURE`: Temperature setting for generation (default: 1.24)
- `MAX_REQUESTS_PER_MINUTE`: Rate limiting (default: 900)

## Deployment to Railway

This repository is configured for easy deployment on Railway:

1. Create a new Railway project
2. Connect to your GitHub repository
3. Railway will automatically detect the Procfile and requirements.txt
4. Add the following environment variables in Railway:
   - `OPENAI_API_KEY` (required)
   - Any other variables from `.env.example` you want to customize

### Railway Quick Deploy

You can also use the Railway CLI for deployment:

```bash
railway login
railway link
railway up
```

## Production Considerations

- Add proper authentication for the API in production
- Use environment variables for all sensitive information
- Consider implementing a more robust rate-limiting solution
- Monitor API usage and costs

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.