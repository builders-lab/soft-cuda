import os
from dotenv import load_dotenv, dotenv_values
from google import genai
from google.genai import types
import sys
load_dotenv()

client = genai.Client()

with open("system_prompt.txt", "r") as f:
    SYSTEM_PROMPT= f.read()

history = sys.stdin.read()


res = client.models.generate_content(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        thinking_config=types.ThinkingConfig(thinking_level="medium")
    ),
    contents=history
)

reply = res.text

# NOTE: we are just printing it we will handle inputing in file in bash

print(reply)
