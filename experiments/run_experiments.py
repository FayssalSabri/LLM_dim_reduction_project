from retrieval.retriever import Retriever
from llm.llm_client import LLMClient
from llm.rag_pipeline import RAGPipeline
import config
import json

# Note : LLM as judge, Format Markdown du prompt

# Init modules
model_name = config.model_name
retriever = Retriever("retrieval/metadata.json")
llm_client = LLMClient(model_name)  # Ollama local
rag_pipeline = RAGPipeline(retriever, llm_client)

# Charger le prompt de base
with open("prompts/context_prompt.txt", "r") as f:
    prompt = f.read()

# Lancer la requête
query = "Recommande-moi les variables à conserver pour le dataset CIC DIAD IoT 2024."

# Appel pipeline (le prompt est traité dans rag_pipeline.py)
features = rag_pipeline.run(query, prompt)

safe_model_name = model_name.replace(".", "_")
with open(f"selected_features_with_{safe_model_name}.json", "w", encoding="utf-8") as f:
    json.dump(features, f, ensure_ascii=False, indent=2)
