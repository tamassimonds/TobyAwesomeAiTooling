import os
import openai
import anthropic
import aiohttp
import json
from together import AsyncTogether
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
import asyncio
import requests




# Load environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
together_api_key = os.getenv('TOGETHER_API_KEY')
writer_api_key = os.getenv('WRITER_API_KEY')
hf_api_token = os.getenv('HF_API_TOKEN')
groq_api_key = os.getenv('GROQ_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
#hf_client = InferenceClient(token=hf_api_token)  # Replace with your Hugging Face token

# Initialize clients
async_together_client = AsyncTogether(api_key=together_api_key)
async_openai_client = AsyncOpenAI(api_key=openai_api_key)

# Add NVIDIA API key
nvidia_api_key = os
        return await query_huggingface()
