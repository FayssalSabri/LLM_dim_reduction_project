import re
import time
import json

class RAGPipeline:

    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    def run(self, query, base_prompt):
        model_name = self.llm_client.model_name
        context = self.retriever.retrieve_context(query)
        prompt = base_prompt.replace("{{feature_list}}", self.retriever.get_feature_list())
        prompt = prompt.replace("{{metadata}}", "/n".join(context))
        prompt = prompt.replace("{{query}}", query)

        # Charger la description des features depuis le README
        with open("data/feature_descriptions.txt", "r", encoding="utf-8") as f:
            feature_descriptions = f.read()
        prompt = prompt.replace("{{feature_descriptions}}", feature_descriptions)

        print("========================================== prompt envoyé au LLM ==========================================")
        print(prompt)
        print("\n")
        print("========================================== Pipline  en cours d'exécution...==========================================")
        print(f"========================================== LLM utilisé: {model_name} ==========================================")

        start = time.time()
        response = self.llm_client.generate(prompt)
        end = time.time()

        # Extraction des features depuis la réponse JSON
        try:
            features_dict = json.loads(response)
            features = list(features_dict.keys())
        except Exception as e:
            print("Erreur lors du parsing JSON:", e)
            features = []

        print("==========================================  LLM-DR ========================================== ")
        print(response)

        # Log complet
        safe_model_name = model_name.replace(".", "_")
        log_filename = f"{safe_model_name}_features.log"
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write("========================================== Log de la pipeline ==========================================\n")
            log_file.write("LLM utilisé: " + model_name + "\n")
            log_file.write("Prompt utilisé:\n" + prompt + "\n\n")

            log_file.write("========================================== LLM-DR ==========================================\n")

            log_file.write("Réponse brute:\n" + str(response) + "\n\n")
            log_file.write("Temps d'inférence: " + str(end - start) + " secondes\n\n")
            log_file.write("Features selectionnés:\n" + str(features) + "\n")
            

        print(f"========================================== Temps d'inférence: {end - start:.2f} secondes ==========================================")
        return features 