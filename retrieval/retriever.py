import json
class Retriever:
    def __init__(self, metadata_file):
        with open(metadata_file, "r",encoding="utf-8") as f:
            self.metadata = json.load(f)

    def get_feature_list(self):
        # self.metadata est un dict, donc on récupère les clés (noms des features)
        return "\n".join([f"- {feature_name}" for feature_name in self.metadata.keys()])

    def retrieve_context(self, query):
        context = []
        for feature_name, feature_info in self.metadata.items():
            desc = (
                f"{feature_name} (type: {feature_info.get('dtype', 'N/A')}), "
                f"missing={feature_info.get('num_missing', 'N/A')}, "
                f"unique={feature_info.get('num_unique', 'N/A')}, "
                f"mean={feature_info.get('mean', 'N/A')}, "
                f"std={feature_info.get('std', 'N/A')},"
                f" min={feature_info.get('min', 'N/A')}, "
                f"max={feature_info.get('max', 'N/A')}, "
            )
            context.append(desc)
        return context
