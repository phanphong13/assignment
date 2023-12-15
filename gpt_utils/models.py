from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()

def get_gpt_llm():
    chat_params = {
        "model": "gpt-3.5-turbo-16k", # Bigger context window
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.5, # To avoid pure copy-pasting from docs lookup
        "max_tokens": 8192
    }
    llm = ChatOpenAI(**chat_params)
    return llm