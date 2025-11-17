import os
import requests
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
url = "https://api.anthropic.com/v1/messages"
headers = {
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}
payload = {
    "model": "claude-haiku-4-5",
    "max_tokens": 256,
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Say your model name and a 1-sentence capability highlight."}]}
    ],
}
resp = requests.post(url, headers=headers, json=payload, timeout=60)
resp.raise_for_status()
print("Status:", resp.status_code)
print("Body:", resp.json())