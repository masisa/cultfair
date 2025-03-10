import os
from pathlib import Path

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data/raw"
DATA_PROCESSED = BASE_DIR / "data/processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Dataset URLs
COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
HOFSTEDE_PATH = DATA_RAW / "hofstede.csv"

# Cultural clustering
CLUSTER_PARAMS = {
    "n_clusters": 2,
    "random_state": 42,
    "pca_components": 2
}

# Model training
MODEL_PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
    "n_splits": 3
}

def setup_directories():
    """Ensures that all required directories exist, with error handling."""
    directories = [DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]
    failed_dirs = []

    for dir in directories:
        try:
            dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            failed_dirs.append((dir, str(e)))

    if failed_dirs:
        print("❌ Error creating the following directories:")
        for dir, error in failed_dirs:
            print(f"   - {dir}: {error}")
        raise SystemExit("🚨 Critical Error: Could not create all required directories. Exiting.")
    
    print("✔️ All required directories have been successfully created or already existed.")

# Run setup_directories() when the script is executed
if __name__ == "__main__":
    setup_directories()
