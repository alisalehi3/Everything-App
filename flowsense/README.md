# FlowSense - Simple Real-Time Focus Tracker (SRTFT)

This project is a demonstration of FlowSense's capabilities, building a simple real-time focus tracker (SRTFT) based on synthetic EEG data. It showcases an end-to-end pipeline from data generation to simulated real-time classification.

## Overview

The SRTFT is designed to:
- **Generate Synthetic EEG Data**: Creates a dataset simulating two cognitive states: 'focused' and 'unfocused', each associated with distinct signal characteristics (e.g., different dominant frequencies) across multiple channels.
- **Preprocess Data**: Loads the raw data and segments it into overlapping windows suitable for analysis.
- **Extract Features**: From each window, calculates features including mean amplitude, variance, and spectral power in specific frequency bands (alpha and beta).
- **Train a Classifier**: Uses the extracted features to train a Logistic Regression model to distinguish between 'focused' and 'unfocused' states.
- **Simulate Real-Time Tracking**: Loads the trained model and processes the synthetic data in chunks, mimicking a live EEG feed. It then prints the predicted cognitive state to the console at regular intervals.
- **Include Basic Tests**: Unit tests verify the functionality of individual components.

This system is built following the principles outlined in the FlowSense Project Design Document (PDR), emphasizing modularity, clarity, and systematic development.

## Project Structure

```
flowsense/
│
├── backend/                # Core processing logic
│   ├── api/                # (Placeholder for future API)
│   ├── models/             # Trained models and model training scripts
│   │   ├── classifier.py   # Trains the focus state classifier
│   │   └── srtft_model.pkl # Saved trained model
│   ├── eeg_preprocessing.py  # EEG data loading and windowing
│   └── feature_extraction.py # Feature calculation from EEG windows
│
├── data/                   # Data storage
│   ├── raw/                # Raw data, including synthetic_eeg_data.csv
│   ├── processed/          # (Placeholder for processed data)
│   └── synthetic/
│       └── generate_eeg.py # Script to generate synthetic EEG data
│
├── frontend/               # (Placeholder for future UI)
│   └── app.py
│
├── tests/                  # Unit tests
│   └── test_srtft_pipeline.py
│
├── docs/                   # (Placeholder for detailed documentation)
│   ├── architecture.md
│   └── user_manual.md
│
├── run.py                  # Main script to run the real-time simulation
├── README.md               # This file
└── project_manifest.yaml   # Project metadata (outside flowsense dir)
```

## Setup & Installation

1.  **Clone the repository** (if applicable).
2.  **Python Version**: Ensure you have Python 3.9+ installed.
3.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  **Install Dependencies**:
    ```bash
    pip install numpy pandas scikit-learn joblib scipy
    ```

## Usage

Ensure you are in the root directory of the project where the `flowsense` directory is located.

1.  **Generate Synthetic EEG Data**:
    ```bash
    python flowsense/data/synthetic/generate_eeg.py
    ```
    This will create `flowsense/data/raw/synthetic_eeg_data.csv`.

2.  **Train the Model**:
    ```bash
    python flowsense/backend/models/classifier.py
    ```
    This will process the synthetic data, train a Logistic Regression model, and save it as `flowsense/backend/models/srtft_model.pkl`. It will also print the test accuracy.

3.  **Run the Real-Time Simulation**:
    ```bash
    python flowsense/run.py
    ```
    This will load the trained model and simulate real-time focus state detection, printing updates to the console. Press `Ctrl+C` to stop.

4.  **Run Unit Tests**:
    ```bash
    python -m unittest flowsense/tests/test_srtft_pipeline.py
    ```

## Dependencies

- Python (3.9+)
- numpy
- pandas
- scikit-learn
- joblib
- scipy

This project was autonomously generated and managed by FlowSense AI (simulated by Jules).
