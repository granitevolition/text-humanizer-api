import os
import sys
import logging
import time
import re
import asyncio
from itertools import cycle
from collections import deque
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import OpenAI API client
from openai import AsyncOpenAI

# Set up logging to both file and console
logging.basicConfig(
    level=logging.DEBUG if os.getenv("LOG_LEVEL") == "DEBUG" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'humanizer.log')),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("humanizer-mcp")

# Initialize the MCP server
mcp = FastMCP("Text Humanizer")

# Read API keys from environment variables or use a single key if provided
api_key_env_var = os.getenv("OPENAI_API_KEY")
openai_api_keys = []

if api_key_env_var:
    openai_api_keys = [api_key_env_var]
else:
    # Check for multiple keys
    for i in range(1, 10):  # Check for up to 9 keys
        key_var = f"OPENAI_API_KEY_{i}"
        if os.getenv(key_var):
            openai_api_keys.append(os.getenv(key_var))

# Fallback to example keys (replace with your keys in production)
if not openai_api_keys:
    logger.warning("No API keys found in environment variables. Using placeholders.")
    # Use placeholder values that won't work - you'll need to set real keys
    openai_api_keys = [
        "sk-placeholder-key-1",
        "sk-placeholder-key-2",
    ]

# Rate limiting variables
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "900"))
api_calls = deque(maxlen=MAX_REQUESTS_PER_MINUTE)

# Prompts and models configuration
rewrite_prompt_templates = {
    22: "Rewrite entirely a straightforward scholar",
    21: "Rewrite entirely a straightforward scholar",
}

system_prompts = {
    22: "You are a straightforward scholar",
    21: "You are a straightforward scholar",
}

# Toggle for selecting best sentence
SELECT_BEST_SENTENCE = os.getenv("SELECT_BEST_SENTENCE", "True").lower() == "true"

# Define models - can be overridden by environment variables
default_model = os.getenv("DEFAULT_MODEL", "ft:gpt-3.5-turbo-0125:personal::9hpCfvVt")
gpt_models = cycle([default_model])

@mcp.tool()
async def echo_text(input_text: str) -> str:
    """
    Simply echo back the text for testing the MCP connection.
    """
    logger.info(f"Echo tool called with: {input_text[:50]}...")
    return input_text

@mcp.tool()
async def humanize_text(input_text: str) -> str:
    """
    Return a humanized (more natural) version of the input text.
    Uses OpenAI's fine-tuned models for high-quality text transformation.
    """
    if not input_text or not input_text.strip():
        return "Error: No input text provided"
    
    logger.info(f"Received text to humanize: {input_text[:50]}...")
    
    try:
        # Break text into sentences
        sentences = extract_sentences(input_text)
        if not sentences:
            logger.warning("No sentences extracted, returning original text")
            return input_text

        logger.info(f"Found {len(sentences)} sentences to process")
        
        # Process each sentence
        rewritten_sentences = []
        for i, sentence in enumerate(sentences):
            logger.info(f"Processing sentence {i+1}/{len(sentences)}")
            rewritten = await process_sentence(
                sentence, 
                temperature=float(os.getenv("TEMPERATURE", "1.24")), 
                frequency_penalty=float(os.getenv("FREQUENCY_PENALTY", "0.2")), 
                presence_penalty=float(os.getenv("PRESENCE_PENALTY", "0.2"))
            )
            rewritten_sentences.append(rewritten)
            
        # Combine sentences
        rewritten_text = " ".join(rewritten_sentences)
        
        # Clean the text using GPT-4
        cleaned_text = await clean_text(input_text, rewritten_text)
        
        logger.info(f"Successfully humanized text. Original: {len(input_text)}, Result: {len(cleaned_text)}")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error in humanize_text: {str(e)}")
        return f"Error: {str(e)}"

def extract_sentences(paragraph):
    """Extract sentences from text, handling special cases like abbreviations."""
    def process_line(line):
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
        special_cases = [
            'et al.', 'etc.', 'e.g.', 'i.e.', 'vs.', 'al.', 'ca.', 'cf.',
            'Fig.', 'fig.', 'Eq.', 'eq.', 'No.', 'no.', 'Vol.', 'vol.',
            'Ch.', 'ch.', 'pp.', 'Ed.', 'ed.', 'Eds.', 'eds.', 'ref.',
            'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Gen.', 'Col.', 'Sgt.',
            'Capt.', 'Lt.', 'Sr.', 'Jr.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.',
            'Inc.', 'Ltd.', 'Co.', 'Corp.', 'LLC', 'LLP',
            'a.m.', 'p.m.', 'A.M.', 'P.M.',
            'St.', 'Ave.', 'Blvd.', 'Rd.', 'Mt.',
            'approx.', 'est.', 'proj.', 'aka.', 'viz.',
        ]

        # Temporarily replace special cases to avoid incorrect splitting
        for i, case in enumerate(special_cases):
            line = line.replace(case, f'__SPECIAL_CASE_{i}__')

        # Split into sentences
        line_sentences = re.split(pattern, line)

        # Restore special cases
        for i, case in enumerate(special_cases):
            line_sentences = [s.replace(f'__SPECIAL_CASE_{i}__', case) for s in line_sentences]

        return [sentence.strip() for sentence in line_sentences if sentence.strip()]

    lines = paragraph.split('\n')
    sentences = []
    for line in lines:
        sentences.extend(process_line(line))
    return sentences

def count_words(sentence):
    """Count words in a sentence."""
    return len(re.findall(r'\w+', sentence))

def clean_response(response, sentence):
    """Clean up the API response."""
    if 'Rewrite:' in response:
        response = response.split('Rewrite:')[0].strip()
    return response if response else sentence

def get_next_openai_api_key():
    """Get the next available OpenAI API key in rotation."""
    api_key = openai_api_keys[0]
    if len(openai_api_keys) > 1:
        openai_api_keys.append(openai_api_keys.pop(0))
    return api_key

async def prompt_api(sentence, temperature, frequency_penalty, presence_penalty, system_prompt_index):
    """Send a prompt to the OpenAI API using fine-tuned models."""
    # Apply rate limiting
    current_time = time.time()
    api_calls.append(current_time)
    
    # Check if we're exceeding rate limits
    old_calls = [t for t in api_calls if current_time - t > 60]
    for old_call in old_calls:
        if old_call in api_calls:
            api_calls.remove(old_call)
            
    if len(api_calls) >= MAX_REQUESTS_PER_MINUTE:
        logger.warning(f"Rate limit reached ({len(api_calls)}/{MAX_REQUESTS_PER_MINUTE}). Waiting...")
        await asyncio.sleep(2)

    # Prepare prompt
    rewrite_prompt_template = rewrite_prompt_templates[system_prompt_index]
    prompt = f"{rewrite_prompt_template}\n\n{sentence}"
    max_tokens = int(count_words(sentence) * 1.75)

    # Get API key and create client
    openai_api_key = get_next_openai_api_key()
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    logger.info(f"Using API key ending in ...{openai_api_key[-5:]}")

    try:
        # Get next model in rotation
        model = next(gpt_models)
        logger.info(f"Using model: {model}")
        
        # Send request to OpenAI API
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompts[system_prompt_index]},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens
        )
        
        # Extract and clean response
        rewritten_sentence = response.choices[0].message.content.strip()
        logger.info(f"API response: {rewritten_sentence[:50]}...")
        return clean_response(rewritten_sentence, sentence)
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return sentence

async def select_best_sentence(original, rewrites):
    """Use GPT-4o-mini to select the best rewritten sentence."""
    prompt = f"""Original sentence: {original}
    Rewritten sentences:
    0. {original}
    {' '.join([f"{i + 1}. {rewrite}" for i, rewrite in enumerate(rewrites)])}

    Select the best rewritten sentence. Must select one. 
    The selected sentence MUST not be an exact replica of the original and the best selected sentence is the one that varies (in style OR/ AND word choice) the most from the original while attempting to maintain its meaning. i.e. always choose the one that is LEAST identical to the original while possessing almost the same meaning and is concise. 
    Reply with just the integer (1, 2, 3, OR 4) of the best sentence. Integer only.
    """

    openai_api_key = get_next_openai_api_key()
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    logger.info(f"Selecting best sentence using GPT-4o-mini")

    try:
        selection_model = os.getenv("SELECTION_MODEL", "gpt-4o-mini") 
        response = await openai_client.chat.completions.create(
            model=selection_model,
            messages=[
                {"role": "system", "content": "You are a sentence selection assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5
        )
        
        # Extract the selected index
        selected_index = int(response.choices[0].message.content.strip())
        logger.info(f"Selection model selected index: {selected_index}")
        return selected_index
    except Exception as e:
        logger.error(f"Error with selection API: {str(e)}")
        return 0  # Return original sentence index on error

async def process_sentence(sentence, temperature=1, frequency_penalty=0.0, presence_penalty=0.0):
    """Process a single sentence through multiple fine-tuned models and select the best."""
    logger.info(f"Processing sentence: {sentence}")

    try:
        # Get rewrites from multiple system prompts
        rewrites = await asyncio.gather(*[
            prompt_api(
                sentence, 
                temperature, 
                frequency_penalty, 
                presence_penalty, 
                system_prompt_index
            )
            for system_prompt_index in system_prompts.keys()
        ])

        for i, rewrite in enumerate(rewrites, 1):
            logger.info(f"Rewrite {i}: {rewrite}")

        # Select the best sentence if enabled
        if SELECT_BEST_SENTENCE:
            all_sentences = [sentence] + rewrites
            best_index = await select_best_sentence(sentence, rewrites)
            best_rewrite = all_sentences[best_index]
            logger.info(f"Selected rewrite: {best_rewrite}")
            return best_rewrite
        else:
            # Otherwise just return the first rewrite
            return rewrites[0] if rewrites else sentence

    except Exception as e:
        logger.error(f"Error processing sentence: {str(e)}")
        return sentence

async def clean_text(original_text, rewritten_text):
    """Clean the rewritten text using GPT-4o."""
    system_prompt = """You are a text comparing assistant. Compare the original text with the rewritten text. 
    You work on the rewritten text (may contain some issues) and only use the original for reference.
    Delete sections in the rewritten text that make no sense or contain non-ASCII characters not in the original. 
    Where in doubt preserve the rewritten text as is. 
    Do not add any new content to the rewritten text nor change the WRITING STYLE nor word choice.
    There is a big penalty if you change words only for style. 
    You must ensure all intext citations are correctly present in the rewritten text.
    You can introduce punctuation and correct capitalization and numbering issues.
    You must return the cleaned rewritten text NOT the original text (Very important).
    Avoid using the sentences in the original text. 
    There is a big penalty for every borrowed word from the original. So avoid at all costs.
    Don't alter the writing style at all."""
    
    prompt = f"Original text:\n{original_text}\n\nRewritten text:\n{rewritten_text}"

    openai_api_key = get_next_openai_api_key()
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    
    cleaning_model = os.getenv("CLEANING_MODEL", "gpt-4o")
    logger.info(f"Cleaning text with {cleaning_model}")
    
    try:
        response = await openai_client.chat.completions.create(
            model=cleaning_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=4000
        )
        
        cleaned_text = response.choices[0].message.content.strip()
        logger.info(f"Cleaned text: {cleaned_text[:50]}...")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return rewritten_text  # Return the uncleaned text on error

# Run the server when the script is executed
if __name__ == "__main__":
    logger.info("Starting Text Humanizer MCP server...")
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    mcp.run(host=host, port=port)