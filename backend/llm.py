# Georgia Kazara
# Reg. No. 20222216
# Thesis Bias Mapper: llm.py
# --------------------------

import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# load API keys from the .env file — keeps credentials out of the code
# --------------------------
load_dotenv()

# OpenAI client
# --------------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Anthropic (Claude) client
# --------------------------
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# send a prompt to OpenAI and return the response text
# --------------------------
def call_openai(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {str(e)}")


# send a prompt to Claude and return the response text
# max_tokens raised to 500 — 300 was cutting responses short for the framing prompts
# --------------------------
def call_claude(prompt):
    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    except Exception as e:
        raise RuntimeError(f"Claude API call failed: {str(e)}")


# unified routing function — choose provider based on model_name
# --------------------------
def call_llm(prompt, model_name):
    model_name = model_name.lower()

    if model_name in ["openai", "gpt-4o-mini", "gpt4o-mini"]:
        return call_openai(prompt)

    if model_name in ["claude", "claude-haiku", "claude-haiku-4-5"]:
        return call_claude(prompt)

    raise ValueError(f"Unsupported model: {model_name}")