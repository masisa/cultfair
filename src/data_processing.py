import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from settings import DATA_RAW, DATA_PROCESSED, COMPAS_URL, HOFSTEDE_PATH, CLUSTER_PARAMS, setup_directories

# Ensure directories exist before running the script
setup_directories()

def load_compas():
    """Load and preprocess the COMPAS dataset."""
    try:
        df = pd.read_csv(COMPAS_URL)
    except Exception as e:
        print(f"❌ Error loading COMPAS dataset: {e}")
        return None

    # Feature selection and cleaning
    features = ["age", "sex", "race", "priors_count", 
               "c_charge_degree", "decile_score", "two_year_recid"]
    
    df = df[features].dropna().rename(columns={
        "sex": "gender",
        "race": "ethnicity",
        "c_charge_degree": "charge_severity",
        "two_year_recid": "recidivism"
    })
    
    # Convert categorical values
    df["charge_severity"] = df["charge_severity"].map({"F": 1, "M": 0})
    
    # Save the cleaned dataset
    try:
        df.to_csv(DATA_RAW / "compas.csv", index=False)
        print("✔️ COMPAS dataset successfully saved.")
    except Exception as e:
        print(f"❌ Error saving COMPAS dataset: {e}")
    
    return df

def process_cultural_dimensions():
    """Process Hofstede's cultural dimensions dataset."""
    
    # Check if Hofstede dataset exists
    if not HOFSTEDE_PATH.exists():
        print(f"❌ Error: Hofstede dataset not found at {HOFSTEDE_PATH}")
        return None

    try:
        hofstede = pd.read_csv(HOFSTEDE_PATH)
    except Exception as e:
        print(f"❌ Error loading Hofstede dataset: {e}")
        return None

    # Apply PCA for cultural dimensions
    pca = PCA(n_components=CLUSTER_PARAMS["pca_components"])
    cultural_pca = pca.fit_transform(hofstede.drop("country", axis=1))
    hofstede[["pca1", "pca2"]] = cultural_pca

    # Apply K-Means clustering
    kmeans = KMeans(
        n_clusters=CLUSTER_PARAMS["n_clusters"],
        random_state=CLUSTER_PARAMS["random_state"]
    )
    hofstede["cluster"] = kmeans.fit_predict(cultural_pca)

    # Save the processed dataset
    try:
        hofstede.to_csv(DATA_PROCESSED / "cultural_clusters.csv", index=False)
        print("✔️ Cultural clusters dataset successfully saved.")
    except Exception as e:
        print(f"❌ Error saving cultural clusters dataset: {e}")

    return hofstede

def prepare_datasets():
    """Main data processing pipeline."""
    compas = load_compas()
    cultural = process_cultural_dimensions()

    # Check if both datasets were successfully processed
    if compas is not None and cultural is not None:
        print("✔️ All datasets have been successfully processed and saved.")
    else:
        print("⚠️ Some datasets could not be processed.")

    return compas, cultural

# Run the script only if executed directly
if __name__ == "__main__":
    prepare_datasets()
