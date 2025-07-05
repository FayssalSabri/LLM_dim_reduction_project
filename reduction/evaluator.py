from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import config

model_name = config.model_names

file_path = "./data/Benign_BruteForce_Mirai_balanced.csv"
from Extract_meta_data import load_data
import pandas as pd
import warnings 
import json
warnings.filterwarnings("ignore")
def main():
    load_data(file_path)
    df = pd.read_csv(file_path)

    

    # Select only numeric columns for features
    X_full = df.drop(columns=['Label'])
    X_full = X_full.select_dtypes(include=['number'])
    # Replace inf/-inf with NaN, then fill NaN with column mean
    X_full = X_full.replace([float('inf'), float('-inf')], pd.NA)
    X_full = X_full.fillna(X_full.mean())
    y = df['Label'].values


    # Feature_LLM = ['Flow ID', 'Src IP', 'Dst IP', 'Protocol', 'Timestamp', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count'] # Mistral (prompt de base)
    # Feature_LLM = ['Flow ID', 'Src IP', 'Dst IP', 'Protocol', 'Timestamp', 'Total Fwd Packet', 'Total Bwd packets', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Total', 'Bwd IAT Total', 'Down/Up Ratio', 'Average Packet Size'] # llama32 (prompt de base)
    safe_model_name = model_name.replace(".", "_")
    with open(f"selected_features_with_{safe_model_name}.json", "r", encoding="utf-8") as f:
        Feature_LLM = json.load(f)

    col_map = {col.lower(): col for col in df.columns}
    Feature_LLM = [col_map[col.lower()] for col in Feature_LLM if col.lower() in col_map and col.lower() != 'Label']
    # Feature_LLM = [col for col in Feature_LLM if col in df.columns]
    print("Features Selected:", Feature_LLM)

    # encode categorical features in Feature_LLM
    for col in Feature_LLM:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]


    # print("Features Selected", Feature_LLM)
    print("\nNbr of Selected Features with LLM:", len(Feature_LLM))  

    X_LLM = df[Feature_LLM]
    X_LLM = X_LLM.replace([float('inf'), float('-inf')], pd.NA)
    X_LLM = X_LLM.fillna(X_LLM.mean())
    X_LLM = X_LLM.values

    # Normalize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)   
    X_LLM = scaler.fit_transform(X_LLM)
    print("Shape of X_full:", X_full.shape) 
    print("Shape of X_LLM:", X_LLM.shape,'\n')  


    X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
    X_train_LLM, X_test_LLM, y_train_LLM, y_test_LLM = train_test_split(X_LLM, y, test_size=0.2, random_state=42)

    model_full = RandomForestClassifier(random_state=42)
    model_LLM = RandomForestClassifier(random_state=42)

    model_full.fit(X_train_full, y_train)
    model_LLM.fit(X_train_LLM, y_train)

    y_pred_full = model_full.predict(X_test_full)
    y_pred_LLM = model_LLM.predict(X_test_LLM)

    print("Performance with all features:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred_full), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred_full, average='weighted'), 4))
    print("ROC AUC:", round(roc_auc_score(y_test, model_full.predict_proba(X_test_full), multi_class='ovr', average='weighted'), 4))

    print("\nPerformance with selected features:")
    print("Accuracy:", round(accuracy_score(y_test_LLM, y_pred_LLM), 4))
    print("F1 Score:", round(f1_score(y_test_LLM, y_pred_LLM, average='weighted'), 4))
    print("ROC AUC:", round(roc_auc_score(y_test_LLM, model_LLM.predict_proba(X_test_LLM), multi_class='ovr', average='weighted'), 4))

    # Note : ajouter le temps d'inférence et d"'entraînement du modèle


    log_filename = f"{safe_model_name}_features.log"
    with open(log_filename, "a") as log_file:
        log_file.write("========================================== Classification Results with Selected Features ==========================================\n")
        # log_file.write("Features Selected: " + str(Feature_LLM) + "\n")
        log_file.write("Nbr of Selected Features with LLM: " + str(len(Feature_LLM)) + "\n")
        log_file.write("Performance with all features:\n")
        log_file.write("Accuracy: " + str(accuracy_score(y_test, y_pred_full)) + "\n")
        log_file.write("F1 Score: " + str(f1_score(y_test, y_pred_full, average='weighted')) + "\n")
        log_file.write("ROC AUC: " + str(roc_auc_score(y_test, model_full.predict_proba(X_test_full), multi_class='ovr', average='weighted')) + "\n")
        log_file.write("\nPerformance with selected features:\n")
        log_file.write("Accuracy: " + str(accuracy_score(y_test_LLM, y_pred_LLM)) + "\n")
        log_file.write("F1 Score: " + str(f1_score(y_test_LLM, y_pred_LLM, average='weighted')) + "\n")
        log_file.write("ROC AUC: " + str(roc_auc_score(y_test_LLM, model_LLM.predict_proba(X_test_LLM), multi_class='ovr', average='weighted')) + "\n\n")

if __name__ == "__main__":
    main()