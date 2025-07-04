import requests

class LLMClient:
    def __init__(self, model_name):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"  # Ollama local

    def generate(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            # "options": {
            #     "temperature": 0.7,
            #     "top_p": 0.9,
            #     "num_predict": 200
            # }
        }
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
