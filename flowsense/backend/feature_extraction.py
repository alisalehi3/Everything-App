import numpy as np
from scipy.fft import rfft, rfftfreq # Using scipy.fft for potentially better performance/options
from typing import Tuple

def calculate_mean_amplitude(windowed_data: np.ndarray) -> np.ndarray:
    """
    Calculates the mean amplitude for each channel in each window.

    Args:
        windowed_data (np.ndarray): 3D array (num_windows, num_channels, window_samples).

    Returns:
        np.ndarray: 2D array (num_windows, num_channels) of mean amplitudes.
    """
    if windowed_data.ndim != 3:
        raise ValueError(f"windowed_data must be a 3D array, got shape {windowed_data.shape}")
    if windowed_data.size == 0:
        return np.array([]).reshape(0, windowed_data.shape[1] if windowed_data.ndim > 1 else 0)
    return np.mean(windowed_data, axis=2)

def calculate_variance(windowed_data: np.ndarray) -> np.ndarray:
    """
    Calculates the variance for each channel in each window.

    Args:
        windowed_data (np.ndarray): 3D array (num_windows, num_channels, window_samples).

    Returns:
        np.ndarray: 2D array (num_windows, num_channels) of variances.
    """
    if windowed_data.ndim != 3:
        raise ValueError(f"windowed_data must be a 3D array, got shape {windowed_data.shape}")
    if windowed_data.size == 0:
        return np.array([]).reshape(0, windowed_data.shape[1] if windowed_data.ndim > 1 else 0)
    return np.var(windowed_data, axis=2)

def calculate_band_power(
    windowed_data: np.ndarray,
    sampling_rate: int,
    low_freq: float,
    high_freq: float
) -> np.ndarray:
    """
    Calculates the power in a specific frequency band for each channel in each window.

    Args:
        windowed_data (np.ndarray): 3D array (num_windows, num_channels, window_samples).
        sampling_rate (int): Sampling rate of the data.
        low_freq (float): Lower bound of the frequency band.
        high_freq (float): Upper bound of the frequency band.

    Returns:
        np.ndarray: 2D array (num_windows, num_channels) of band powers.
    """
    if windowed_data.ndim != 3:
        raise ValueError(f"windowed_data must be a 3D array, got shape {windowed_data.shape}")
    if windowed_data.size == 0:
        return np.array([]).reshape(0, windowed_data.shape[1] if windowed_data.ndim > 1 else 0)

    num_windows, num_channels, num_samples = windowed_data.shape
    band_powers = np.zeros((num_windows, num_channels))

    # Get frequencies for the FFT
    freqs = rfftfreq(num_samples, 1.0 / sampling_rate)
    freq_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]

    if len(freq_indices) == 0:
        print(f"Warning: No frequency bins found for band {low_freq}-{high_freq} Hz. Returning zero power.")
        return band_powers

    for i in range(num_windows):
        for j in range(num_channels):
            signal_window = windowed_data[i, j, :]
            fft_coeffs = rfft(signal_window)
            psd = np.abs(fft_coeffs)**2
            band_powers[i, j] = np.sum(psd[freq_indices]) / len(freq_indices) # Average power in band

    return band_powers

def extract_features(windowed_data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Extracts a set of features from windowed EEG data.

    Args:
        windowed_data (np.ndarray): 3D array (num_windows, num_channels, window_samples).
        sampling_rate (int): Sampling rate of the data.

    Returns:
        np.ndarray: 2D array (num_windows, num_features) containing concatenated features.
                    Features are ordered as: [mean_ch1, mean_ch2..., var_ch1, var_ch2...,
                                             alpha_ch1, alpha_ch2..., beta_ch1, beta_ch2...]
    """
    if windowed_data.ndim != 3 or windowed_data.size == 0 :
        # Return an empty array with expected number of feature columns if input is empty
        # Assuming 4 types of features (mean, var, alpha, beta) per channel
        num_channels_guess = windowed_data.shape[1] if windowed_data.ndim == 3 and windowed_data.shape[1] > 0 else 1 # Guess 1 channel if unknown
        return np.array([]).reshape(0, num_channels_guess * 4)


    mean_amp_features = calculate_mean_amplitude(windowed_data)
    variance_features = calculate_variance(windowed_data)

    # Define frequency bands based on synthetic data generation
    alpha_band_power = calculate_band_power(windowed_data, sampling_rate, low_freq=8.0, high_freq=12.0) # Focused state freq = 10Hz
    beta_band_power = calculate_band_power(windowed_data, sampling_rate, low_freq=15.0, high_freq=30.0) # Unfocused state freq = 30Hz

    all_features = [
        mean_amp_features,
        variance_features,
        alpha_band_power,
        beta_band_power
    ]

    # Concatenate along the features axis (axis=1)
    concatenated_features = np.concatenate(all_features, axis=1)

    print(f"Extracted features. Shape: {concatenated_features.shape}")
    return concatenated_features

if __name__ == '__main__':
    # Example Usage
    print("Running example usage for feature_extraction.py...")
    # Requires eeg_preprocessing.py and synthetic data to be available
    # This is for demonstration and basic testing.

    # Attempt to import preprocessing functions
    try:
        from flowsense.backend.eeg_preprocessing import load_eeg_data, create_windows
        preprocessing_available = True
    except ImportError:
        print("Could not import from flowsense.backend.eeg_preprocessing. Make sure it's in the PYTHONPATH.")
        preprocessing_available = False

    if preprocessing_available:
        example_data_path = 'flowsense/data/raw/synthetic_eeg_data.csv'
        sampling_rate_param = 250  # Must match the generation script

        import os
        if not os.path.exists(example_data_path):
            print(f"Example data file not found: {example_data_path}")
            print("Please run `flowsense/data/synthetic/generate_eeg.py` first.")
        else:
            raw_data_df = load_eeg_data(example_data_path)
            if not raw_data_df.empty:
                windows, _ = create_windows(raw_data_df,
                                            window_size_seconds=2.0,
                                            overlap_seconds=1.0,
                                            sampling_rate=sampling_rate_param)

                if windows.size > 0:
                    features = extract_features(windows, sampling_rate_param)
                    if features.size > 0:
                        print(f"Successfully extracted features. Example first feature vector: {features[0,:]}")
                    else:
                        print("Feature extraction resulted in an empty feature set.")
                else:
                    print("Window creation resulted in no windows, skipping feature extraction example.")
