import pandas as pd
import numpy as np
import joblib
import time
import os
import sys

# Adjust PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # We will process raw data chunks directly for a more "realistic" simulation
    # instead of re-using create_windows which expects a full dataset.
    from flowsense.backend.feature_extraction import extract_features
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure PYTHONPATH is set correctly or run from project root.")
    sys.exit(1)

# Configuration
MODEL_PATH = 'flowsense/backend/models/srtft_model.pkl'
DATA_PATH = 'flowsense/data/raw/synthetic_eeg_data.csv' # Full raw data
SAMPLING_RATE = 250  # Hz
WINDOW_SIZE_SECONDS = 2.0  # Duration of data for one prediction
SIMULATION_INTERVAL_SECONDS = 2.0 # How often to run the loop
NUM_CHANNELS = 4 # As defined in data generation
CHANNEL_COLUMNS = [f'ch{i+1}' for i in range(NUM_CHANNELS)]

def preprocess_live_window(raw_window_df: pd.DataFrame, expected_samples: int) -> np.ndarray:
    """
    Prepares a raw data window (DataFrame segment) into the numpy format
    expected by extract_features.
    Handles potential padding or truncation if the segment isn't exact.
    Output shape: (1, num_channels, expected_samples)
    """

    # Ensure all channel columns are present
    if not all(col in raw_window_df.columns for col in CHANNEL_COLUMNS):
        raise ValueError(f"Missing one or more channel columns ({CHANNEL_COLUMNS}) in provided data.")

    # Extract channel data and transpose to (num_channels, num_samples_in_chunk)
    live_eeg_data = raw_window_df[CHANNEL_COLUMNS].values.T

    current_samples = live_eeg_data.shape[1]

    if current_samples == expected_samples:
        processed_window = live_eeg_data
    elif current_samples > expected_samples: # Truncate if too long
        processed_window = live_eeg_data[:, :expected_samples]
    else: # Pad with zeros if too short
        padding = np.zeros((NUM_CHANNELS, expected_samples - current_samples))
        processed_window = np.hstack((live_eeg_data, padding))

    return processed_window.reshape(1, NUM_CHANNELS, expected_samples)


def main_simulation_loop():
    """
    Main loop to simulate real-time EEG processing and focus state detection.
    """
    print("Starting Simple Real-Time Focus Tracker (SRTFT) Simulation...")

    # 1. Load Model
    print(f"Loading trained model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
        return
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print("Model loaded successfully.")

    # 2. Load Raw EEG Data (or a reference to it)
    print(f"Loading raw EEG data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}. Please generate data first.")
        return
    try:
        full_raw_data_df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print(f"Data loaded. Total samples: {len(full_raw_data_df)}")

    # Simulation parameters
    samples_per_window = int(WINDOW_SIZE_SECONDS * SAMPLING_RATE) # Samples in a 2s window
    samples_per_step = int(SIMULATION_INTERVAL_SECONDS * SAMPLING_RATE) # Advance by 2s of data

    current_pos = 0
    simulated_time_seconds = 0.0

    print(f"\nStarting simulation loop (Interval: {SIMULATION_INTERVAL_SECONDS}s, Window: {WINDOW_SIZE_SECONDS}s)...")
    print("Press Ctrl+C to stop.\n")

    try:
        while current_pos + samples_per_window <= len(full_raw_data_df):
            # Get the current chunk of raw data for this window
            raw_data_chunk_df = full_raw_data_df.iloc[current_pos : current_pos + samples_per_window]

            if raw_data_chunk_df.empty:
                print("End of data stream reached (empty chunk).")
                break

            # Preprocess this "live" window
            # This prepares a (1, num_channels, samples_per_window) numpy array
            live_window_np = preprocess_live_window(raw_data_chunk_df, samples_per_window)

            # Extract features. extract_features expects (num_windows, num_channels, window_samples)
            # So, live_window_np is already in the correct shape for a single window.
            current_features = extract_features(live_window_np, SAMPLING_RATE) # Should be (1, num_features)

            if current_features.size == 0:
                print("Feature extraction failed for the current window.")
                current_pos += samples_per_step
                simulated_time_seconds += SIMULATION_INTERVAL_SECONDS
                time.sleep(SIMULATION_INTERVAL_SECONDS)
                continue

            # Predict state
            predicted_label = model.predict(current_features)[0] # model.predict returns an array
            predicted_proba = model.predict_proba(current_features)[0] # Probabilities for [class_0, class_1]

            state = "Focused" if predicted_label == 1 else "Unfocused"
            confidence = predicted_proba[int(predicted_label)]

            # Output to console
            actual_timestamp = raw_data_chunk_df['timestamp'].iloc[-1] # Timestamp of last sample in window
            print(f"Time: {actual_timestamp:.2f}s | State: {state} (Confidence: {confidence:.2f}) | Raw Label: {raw_data_chunk_df['label'].iloc[-1]}")

            # Move to the next segment of data
            current_pos += samples_per_step
            simulated_time_seconds += SIMULATION_INTERVAL_SECONDS

            time.sleep(SIMULATION_INTERVAL_SECONDS)

        print("\nEnd of dataset reached or simulation stopped.")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_simulation_loop()
