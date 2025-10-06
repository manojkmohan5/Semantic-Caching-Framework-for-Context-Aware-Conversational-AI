import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

class GeminiClient:
    def __init__(self):
        self.generation_model = "gemini-2.5-pro"
        self.embedding_model = "text-embedding-004"

    def get_embedding(self, text):
        """Get text embedding from Gemini API."""
        url = (
            f"https://generativelanguage.googleapis.com/v1/models/"
            f"{self.embedding_model}:embedContent?key={API_KEY}"
        )
        payload = { "content": { "parts": [ { "text": text } ] } }
        resp = requests.post(url, json=payload)
        if resp.ok:
            data = resp.json()
            # Returns the first embedding vector in the response
            return data["embedding"]["values"]
        else:
            print("Embedding API call failed!")
            print("Status code:", resp.status_code)
            print("Response:", resp.text)
            return None

    def call_llm(self, prompt):
        """Generate text from Gemini."""
        url = (
            f"https://generativelanguage.googleapis.com/v1/models/"
            f"{self.generation_model}:generateContent?key={API_KEY}"
        )
        payload = {
            "contents": [ { "parts": [ { "text": prompt } ] } ]
        }
        resp = requests.post(url, json=payload)
        if resp.ok:
            data = resp.json()
            return (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "No answer.")
            )
        else:
            print("LLM API call failed!")
            print("Status code:", resp.status_code)
            print("Response:", resp.text)
            return "Sorry, generation failed."
