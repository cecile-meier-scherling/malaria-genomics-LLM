import requests
import os

# Path to data table
DATA_PATH = "data/raw/all_who_get_prevalence.csv"  # or .parquet

# Deine LLM Model
LLM_MODEL = "llama3"

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call a local Ollama model running at http://localhost:11434.
    Uses the /api/chat endpoint with a system + user message.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    resp = requests.post(url, json=payload)

    if resp.status_code != 200:
        # Print Ollama's error message for debugging
        print("Ollama error:", resp.status_code, resp.text)
        resp.raise_for_status()

    data = resp.json()
    return data["message"]["content"].strip()