import unittest
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Adjust PYTHONPATH to find the 'flowsense' package
# Assumes tests are in flowsense/tests/ and project root is two levels up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Conditional imports for modules to be tested
try:
    from flowsense.data.synthetic.generate_eeg import main as generate_eeg_main, OUTPUT_DIR as EEG_OUTPUT_DIR, OUTPUT_FILENAME as EEG_FILENAME
    from flowsense.backend.eeg_preprocessing import load_eeg_data, create_windows
    from flowsense.backend.feature_extraction import extract_features, calculate_mean_amplitude, calculate_variance, calculate_band_power
    # For model loading, we'll directly use joblib and check path from classifier constants
    from flowsense.backend.models.classifier import MODEL_OUTPUT_DIR as CLF_MODEL_DIR, MODEL_FILENAME as CLF_MODEL_FILENAME
except ImportError as e:
    print(f"Failed to import modules for testing: {e}")
    print("Ensure you run tests from the project root or that PYTHONPATH is correctly set.")
    # We'll let tests fail individually if modules aren't found, to pinpoint issues.
    pass

# Constants for tests
SAMPLING_RATE = 250
NUM_CHANNELS_SYNTHETIC = 4 # From generate_eeg script
EXPECTED_EEG_COLUMNS = ['timestamp', 'ch1', 'ch2', 'ch3', 'ch4', 'label']
SYNTHETIC_DATA_PATH = os.path.join(EEG_OUTPUT_DIR, EEG_FILENAME) # This will be flowsense/data/raw/synthetic_eeg_data.csv
MODEL_PATH_FOR_TEST = os.path.join(CLF_MODEL_DIR, CLF_MODEL_FILENAME) # This will be flowsense/backend/models/srtft_model.pkl


class TestSRTFTPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure synthetic data and model exist for tests that need them."""
        # Construct absolute paths based on project_root for resources
        cls.synthetic_data_full_path = os.path.join(project_root, SYNTHETIC_DATA_PATH)
        cls.model_full_path = os.path.join(project_root, MODEL_PATH_FOR_TEST)
        eeg_output_dir_full_path = os.path.join(project_root, EEG_OUTPUT_DIR)

        # Check for synthetic data
        if not os.path.exists(cls.synthetic_data_full_path):
            print(f"Synthetic data not found at {cls.synthetic_data_full_path}. Attempting to generate...")
            try:
                # Ensure the output directory for EEG data exists before generating
                os.makedirs(eeg_output_dir_full_path, exist_ok=True)
                generate_eeg_main() # This function prints success/failure
            except Exception as e:
                print(f"Could not generate synthetic EEG data for tests: {e}")

        # Model existence is checked in its specific test, as it's a product of another script run.

    def test_01_synthetic_data_generation_file_exists(self):
        """Test if synthetic EEG data file is created."""
        self.assertTrue(os.path.exists(self.synthetic_data_full_path), f"Synthetic EEG data file not found: {self.synthetic_data_full_path}")

    def test_02_synthetic_data_columns(self):
        """Test if the synthetic EEG data has expected columns."""
        if not os.path.exists(self.synthetic_data_full_path):
            self.skipTest(f"Skipping column check as data file {self.synthetic_data_full_path} is missing.")

        df = pd.read_csv(self.synthetic_data_full_path)
        self.assertListEqual(list(df.columns), EXPECTED_EEG_COLUMNS, "CSV columns mismatch.")
        self.assertTrue(len(df) > 0, "Synthetic data CSV is empty.")

    def test_03_load_eeg_data_preprocessing(self):
        """Test loading EEG data functionality from eeg_preprocessing."""
        dummy_csv_data = "timestamp,ch1,ch2,label\n0.0,1,10,0\n0.004,2,12,0"
        # Create dummy file in a location relative to project root to ensure write access
        dummy_filepath = os.path.join(project_root, "dummy_eeg_for_test.csv")

        with open(dummy_filepath, 'w') as f:
            f.write(dummy_csv_data)

        df = load_eeg_data(dummy_filepath)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        os.remove(dummy_filepath) # Clean up

    def test_04_create_windows_preprocessing(self):
        """Test window creation functionality from eeg_preprocessing."""
        data = {
            'timestamp': np.linspace(0, 9.996, 10 * SAMPLING_RATE), # 10 seconds of data, end point adjusted for linspace
            'ch1': np.random.rand(10 * SAMPLING_RATE),
            'ch2': np.random.rand(10 * SAMPLING_RATE),
            'label': np.random.randint(0, 2, 10 * SAMPLING_RATE)
        }
        df = pd.DataFrame(data)

        window_size_sec = 2.0
        overlap_sec = 1.0
        expected_window_samples = int(window_size_sec * SAMPLING_RATE)

        windows, labels = create_windows(df, window_size_sec, overlap_sec, SAMPLING_RATE, channel_columns=['ch1', 'ch2'])

        self.assertEqual(windows.ndim, 3, "Windowed data should be 3D.")
        self.assertTrue(windows.shape[0] > 0, "Number of windows should be greater than 0.")
        self.assertEqual(windows.shape[1], 2, "Number of channels in windowed data mismatch.")
        self.assertEqual(windows.shape[2], expected_window_samples, "Samples per window mismatch.")
        self.assertEqual(len(labels), windows.shape[0], "Mismatch between number of windows and labels.")

    def test_05_feature_extraction_shapes(self):
        """Test feature extraction output shape."""
        num_windows = 5
        num_channels = 2
        window_samples = int(2.0 * SAMPLING_RATE) # 2s windows
        dummy_windowed_data = np.random.rand(num_windows, num_channels, window_samples)

        features = extract_features(dummy_windowed_data, SAMPLING_RATE)

        self.assertEqual(features.ndim, 2, "Features array should be 2D.")
        self.assertEqual(features.shape[0], num_windows, "Number of windows in features mismatch.")
        self.assertEqual(features.shape[1], num_channels * 4, "Number of features mismatch.")

    def test_06_basic_feature_calculations(self):
        """Test individual feature calculations for sanity checks."""
        dummy_window = np.array([[[1., 2., 3., 4., 5.], [10., 10., 10., 10., 10.]]]) # 1 window, 2 channels, 5 samples

        mean_amp = calculate_mean_amplitude(dummy_window)
        self.assertAlmostEqual(mean_amp[0,0], 3.0)
        self.assertAlmostEqual(mean_amp[0,1], 10.0)

        variance = calculate_variance(dummy_window)
        self.assertAlmostEqual(variance[0,0], 2.0) # Var of [1,2,3,4,5] is 2.0
        self.assertAlmostEqual(variance[0,1], 0.0) # Var of [10,10,10,10,10]

        # Basic band power check (sampling rate must be high enough for FFT freq bins)
        # For 5 samples, max freq is SAMPLING_RATE/2. Freq bins are 0, SR/N, 2SR/N ...
        # N=5, SR=250. Freqs: 0, 50, 100. rfftfreq(5, 1/250) -> [  0.,  50., 100.]
        # If band 8-12Hz is requested, it will likely find no bins.
        # Let's use a band that will have a bin, e.g. 40-60 Hz for the 50Hz bin
        power = calculate_band_power(dummy_window, SAMPLING_RATE, 40, 60) # tests 50Hz bin
        self.assertEqual(power.shape, (1,2))


    def test_07_model_loading_and_prediction(self):
        """Test if the trained model can be loaded and make a prediction."""
        if not os.path.exists(self.model_full_path):
            self.skipTest(f"Skipping model test as model file {self.model_full_path} is missing. Run classifier.py first.")

        model = joblib.load(self.model_full_path)
        self.assertIsNotNone(model, "Failed to load the model.")

        num_expected_features = NUM_CHANNELS_SYNTHETIC * 4
        dummy_feature_vector = np.random.rand(1, num_expected_features)

        try:
            prediction = model.predict(dummy_feature_vector)
            proba = model.predict_proba(dummy_feature_vector)
            self.assertTrue(prediction is not None, "Model prediction returned None.")
            self.assertEqual(prediction.ndim, 1, "Prediction should be a 1D array.")
            self.assertEqual(proba.shape, (1,2), "Probabilities shape incorrect.")
        except Exception as e:
            self.fail(f"Model prediction failed with error: {e}")

if __name__ == '__main__':
    # This allows running the tests directly from the script file
    # while ensuring the test runner can find them.
    unittest.main()
