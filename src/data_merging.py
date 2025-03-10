import pandas as pd
import numpy as np
from settings import DATA_PROCESSED, MODEL_PARAMS, DATA_RAW
from sklearn.model_selection import train_test_split

CULTURAL_MAPPING = {

    "African-American": ["Nigeria", "Ghana"],  # West Africa as the main region
    "Caucasian": ["Germany", "USA", "Canada", "Australia", "Norway", "Turkey"],  # Turkey has historical ties to Europe
    "Hispanic": ["Mexico", "Brazil", "Spain"],  # Spanish-/Portuguese-speaking countries
    "Asian": ["China", "Japan", "India", "Indonesia", "Iran"],  # Iran shares some cultural traits with South Asia
    "Native American": ["USA", "Mexico"],  # Mainly indigenous groups in North America
    "Other": ["Russia", "Norway", "Indonesia", "Saudi Arabia"]  # Saudi Arabia fits here due to lack of a direct match

}

def merge_data():
    """Merges COMPAS and cultural data, assigns cultural clusters, and splits the data into training and test sets."""
    
    # Load datasets
    compas = pd.read_csv(DATA_RAW / "compas.csv")
    cultural = pd.read_csv(DATA_PROCESSED / "cultural_clusters.csv")
    
    # Map ethnicity to country based on the cultural mapping
    compas["country"] = compas["ethnicity"].map(
        lambda x: np.random.choice(CULTURAL_MAPPING.get(x, ["USA"]))
    )
    
    # Merge COMPAS data with cultural clusters
    merged = compas.merge(
        cultural[["country", "cluster", "pca1", "pca2"]],
        on="country",
        how="left"
    )
    
    # Split data into training and test sets
    train, test = train_test_split(
        merged,
        test_size=MODEL_PARAMS["test_size"],
        stratify=merged[["cluster", "recidivism"]],
        random_state=MODEL_PARAMS["random_state"]
    )
    
    # Save processed data
    train.to_csv(DATA_PROCESSED / "train.csv", index=False)
    test.to_csv(DATA_PROCESSED / "test.csv", index=False)

    # Success message
    print(f"✅ Data merging and splitting completed successfully! \n"
          f"- Total records: {len(merged)} \n"
          f"- Training set: {len(train)} records \n"
          f"- Test set: {len(test)} records \n"
          f"Processed data saved in: {DATA_PROCESSED}")
    
    return merged

if __name__ == "__main__":
    merged_data = merge_data()
