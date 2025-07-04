import pandas as pd
def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def main():
    # Example usage
    file_path = "data/Benign_BruteForce_Mirai_balanced.csv"  # Replace with your actual file path
    df = load_data(file_path)
    from pandas.api.types import CategoricalDtype

    if df is not None:
        print("Data loaded successfully:")
        print(df.head())
        
        print("\nData types of each column:")
        print(df.dtypes)

        matadata = df.describe(include='all')
        print("\Statistiques descriptives classiques (pour numériques):")    
        print(df.describe().transpose())
        print(df.nunique())
        df_numeric = df.select_dtypes(include=['number'])
        print(df_numeric.corr())
        from scipy.stats import skew, kurtosis

        skewness = df.skew(numeric_only=True)
        kurt = df.kurtosis(numeric_only=True)
        print("Skewness:\n", skewness)
        print("Kurtosis:\n", kurt)
        metadata = {}

        for col in df.columns:
            col_data = df[col]
            meta = {}
            meta['dtype'] = str(col_data.dtype)
            meta['num_missing'] = int(col_data.isna().sum())
            meta['num_unique'] = int(col_data.nunique())
            
            if pd.api.types.is_numeric_dtype(col_data):
                meta['min'] = float(col_data.min())
                meta['max'] = float(col_data.max())
                meta['mean'] = float(col_data.mean())
                meta['std'] = float(col_data.std())
                meta['variance'] = float(col_data.var())
                meta['skewness'] = float(skew(col_data.dropna()))
                meta['kurtosis'] = float(kurtosis(col_data.dropna()))
            elif isinstance(col_data.dtype, CategoricalDtype) or col_data.dtype == 'object':
                meta['top_values'] = col_data.value_counts().head(5).to_dict()
            metadata[col] = meta

        import json
        print(json.dumps(metadata, indent=2))
        # Save metadata to a JSON file
        with open('retrieval/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()