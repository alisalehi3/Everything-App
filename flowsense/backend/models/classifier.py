import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

# Adjust PYTHONPATH to include the project root for sibling module imports
# This assumes the script is in flowsense/backend/models/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flowsense.backend.eeg_preprocessing import load_eeg_data, create_windows
    from flowsense.backend.feature_extraction import extract_features
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure PYTHONPATH is set correctly or run from project root.")
    sys.exit(1)

# Configuration
DATA_FILE_PATH = 'flowsense/data/raw/synthetic_eeg_data.csv'
MODEL_OUTPUT_DIR = 'flowsense/backend/models/'
MODEL_FILENAME = 'srtft_model.pkl'
SAMPLING_RATE = 250  # Hz, must match data generation
WINDOW_SIZE_SECONDS = 2.0
OVERLAP_SECONDS = 1.0
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

def train_and_evaluate_model(
    data_path: str,
    model_save_path: str,
    sampling_rate: int,
    window_size: float,
    overlap: float,
    test_size: float = TEST_SPLIT_SIZE,
    random_state: int = RANDOM_SEED
) -> dict:
    """
    Loads data, preprocesses, extracts features, trains a classifier,
    evaluates it, and saves the trained model.

    Args:
        data_path (str): Path to the raw EEG data CSV file.
        model_save_path (str): Path where the trained model should be saved.
        sampling_rate (int): Sampling rate of the EEG data.
        window_size (float): Window size in seconds for feature extraction.
        overlap (float): Overlap in seconds for windowing.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing evaluation metrics (e.g., accuracy)
              and the path to the saved model.
    """
    print("Starting model training process...")

    # 1. Load Data
    print(f"Loading data from {data_path}...")
    raw_data_df = load_eeg_data(data_path)
    if raw_data_df.empty:
        print("Loaded data is empty. Exiting.")
        return {"error": "Loaded data is empty."}

    # 2. Create Windows
    print("Creating windows...")
    windowed_signals, window_labels = create_windows(
        raw_data_df,
        window_size_seconds=window_size,
        overlap_seconds=overlap,
        sampling_rate=sampling_rate
    )
    if windowed_signals.size == 0:
        print("No windows created from data. Exiting.")
        return {"error": "Window creation failed or resulted in no windows."}

    # 3. Extract Features
    print("Extracting features...")
    features = extract_features(windowed_signals, sampling_rate)
    if features.size == 0:
        print("No features extracted. Exiting.")
        return {"error": "Feature extraction failed or resulted in no features."}

    print(f"Shape of features: {features.shape}, Shape of labels: {window_labels.shape}")

    # 4. Split Data
    print(f"Splitting data into train/test sets (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, window_labels, test_size=test_size, random_state=random_state, stratify=window_labels
    )
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # 5. Train Model
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=random_state, max_iter=1000) # Increased max_iter for convergence
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 6. Evaluate Model
    print("Evaluating model on the test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save Model
    print(f"Saving trained model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved successfully to {model_save_path}")

    return {
        "accuracy": accuracy,
        "model_path": model_save_path,
        "num_features": features.shape[1],
        "num_train_samples": X_train.shape[0],
        "num_test_samples": X_test.shape[0]
    }

if __name__ == "__main__":
    print("Running Model Training and Evaluation Script...")

    # Ensure the output directory for the model exists
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)

    full_model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)

    training_results = train_and_evaluate_model(
        data_path=DATA_FILE_PATH,
        model_save_path=full_model_path,
        sampling_rate=SAMPLING_RATE,
        window_size=WINDOW_SIZE_SECONDS,
        overlap=OVERLAP_SECONDS
    )

    print("\nTraining process finished.")
    if "error" in training_results:
        print(f"An error occurred: {training_results['error']}")
    else:
        print(f"Final accuracy: {training_results.get('accuracy', 'N/A'):.4f}")
        print(f"Model stored at: {training_results.get('model_path', 'N/A')}")
