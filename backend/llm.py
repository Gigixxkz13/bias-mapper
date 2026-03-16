# Georgia Kazara
# Reg. No. 20222216 
# Thesis Bias Mapper: llm.py
# --------------------------
import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from the .env file
# This allows us to keep API keys outside the code for security
load_dotenv()


# OpenAI client configuration
# ---------------------------
# Create an OpenAI client using the API key stored in the environment variables
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Claude (Anthropic) configuration
# --------------------------------
# Create a Claude client using the Anthropic API key
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Function to call OpenAI
# -----------------------
def call_openai(prompt):

    # Send the prompt to the OpenAI Chat Completions API
    # The system currently uses the lightweight GPT-4o-mini model
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7  # Controls randomness of the output
    )

    # Extract the generated text from the API response
    return response.choices[0].message.content

# Function to call Claude
# -----------------------
def call_claude(prompt):

    # Send the prompt to Anthropic's Claude model
    # Claude Haiku 4.5 is the current lightweight Claude model
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Claude responses are stored slightly differently,
    # so we extract the text from the first content block
    return response.content[0].text

# Unified LLM call interface
# --------------------------
def call_llm(prompt, model_name):

    # Normalize the model name to avoid case issues
    model_name = model_name.lower()

    # If the requested model is OpenAI
    if model_name in ["openai", "gpt-4o-mini", "gpt4o-mini"]:
        return call_openai(prompt)

    # If the requested model is Claude
    if model_name in ["claude", "claude-haiku", "claude-haiku-4-5"]:
        return call_claude(prompt)

    # If the model is not supported, raise an error
    raise ValueError(f"Unsupported model: {model_name}")