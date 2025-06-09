import pandas as pd
import numpy as np
from typing import Tuple

def load_eeg_data(filepath: str) -> pd.DataFrame:
    """
    Loads EEG data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the EEG data.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_windows(
    data: pd.DataFrame,
    window_size_seconds: float,
    overlap_seconds: float,
    sampling_rate: int,
    channel_columns: list = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates overlapping windows from EEG data.

    Args:
        data (pd.DataFrame): DataFrame containing EEG data with 'timestamp', channel columns, and 'label'.
        window_size_seconds (float): Desired window size in seconds.
        overlap_seconds (float): Desired overlap between windows in seconds.
        sampling_rate (int): The sampling rate of the EEG data in Hz.
        channel_columns (list, optional): List of column names for EEG channels.
                                         If None, attempts to auto-detect 'chX' columns.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A 3D numpy array of windowed data (num_windows, num_channels, window_samples).
            - A 1D numpy array of corresponding labels (num_windows).
    """
    if data.empty:
        print("Warning: Input data for windowing is empty.")
        return np.array([]).reshape(0,0,0), np.array([])

    if channel_columns is None:
        channel_columns = [col for col in data.columns if col.startswith('ch')]

    if not channel_columns:
        raise ValueError("No channel columns found. Please specify channel_columns or ensure they are named like 'chX'.")

    eeg_signals = data[channel_columns].values.T  # Shape: (num_channels, num_samples_total)
    labels = data['label'].values

    window_samples = int(window_size_seconds * sampling_rate)
    overlap_samples = int(overlap_seconds * sampling_rate)
    step_samples = window_samples - overlap_samples

    if step_samples <= 0:
        raise ValueError("Window size must be greater than overlap size. Ensure step_samples > 0.")

    num_total_samples = eeg_signals.shape[1]
    num_windows = (num_total_samples - window_samples) // step_samples + 1

    if num_windows <= 0:
        print("Warning: No windows could be created with the given parameters. Data might be too short.")
        return np.array([]).reshape(0,len(channel_columns),window_samples), np.array([])

    windowed_eeg_data = np.zeros((num_windows, eeg_signals.shape[0], window_samples))
    window_labels = np.zeros(num_windows)

    for i in range(num_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples

        windowed_eeg_data[i] = eeg_signals[:, start_idx:end_idx]

        # Label for the window: using the label at the end of the window
        # This is a common approach. Another is mode if labels can change within a window.
        window_labels[i] = labels[end_idx - 1]

    print(f"Created {num_windows} windows. Window shape: {windowed_eeg_data.shape}, Labels shape: {window_labels.shape}")
    return windowed_eeg_data, window_labels

if __name__ == '__main__':
    # Example Usage (assuming the synthetic data script has been run)
    # This part is for demonstration and basic testing when running the script directly.
    # It won't be part of the actual pipeline execution by run.py but helps in development.
    print("Running example usage for eeg_preprocessing.py...")
    example_data_path = 'flowsense/data/raw/synthetic_eeg_data.csv'
    sampling_rate_param = 250  # Must match the generation script

    # Check if example data exists
    import os
    if not os.path.exists(example_data_path):
        print(f"Example data file not found: {example_data_path}")
        print("Please run `flowsense/data/synthetic/generate_eeg.py` first to create the data.")
    else:
        raw_data_df = load_eeg_data(example_data_path)

        if not raw_data_df.empty:
            # Example: 2-second windows, 1-second overlap
            windows, win_labels = create_windows(raw_data_df,
                                                 window_size_seconds=2.0,
                                                 overlap_seconds=1.0,
                                                 sampling_rate=sampling_rate_param)

            if windows.size > 0:
                print(f"Successfully created windows. Example first window data shape: {windows[0].shape}")
                print(f"First few labels: {win_labels[:5]}")
            else:
                print("Window creation resulted in no windows.")
