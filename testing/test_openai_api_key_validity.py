import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

try:
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use a simpler model for testing
        messages=[{"role": "user", "content": "Hello, world!"}],
        max_tokens=5
    )
    print("API key is valid!")
    print(response)
except Exception as e:
    print(f"API key error: {str(e)}")