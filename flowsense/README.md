# FlowSense - Simple Real-Time Focus Tracker (SRTFT)

This project is a demonstration of FlowSense's capabilities, building a simple real-time focus tracker (SRTFT) based on synthetic EEG data. It showcases an end-to-end pipeline from data generation to simulated real-time classification, with options for console-based simulation, a web-based UI, and Docker containerization.

## Overview

The SRTFT is designed to:
- **Generate Synthetic EEG Data**: Creates a dataset simulating two cognitive states: 'focused' and 'unfocused'.
- **Preprocess Data**: Loads raw data and segments it into overlapping windows.
- **Extract Features**: Calculates features like mean amplitude, variance, and spectral power.
- **Train a Classifier**: Uses a Logistic Regression model to distinguish between states.
- **Simulate Real-Time Tracking (Console)**: Provides console feedback on predicted states from data chunks.
- **Provide Web Application (API & UI)**: Offers a FastAPI backend for predictions and a Streamlit frontend for interaction.
- **Enable Dockerization**: Allows packaging the web application into a Docker container.
- **Include Basic Tests**: Unit tests verify core component functionality.

This system is built following the principles outlined in the FlowSense Project Design Document (PDR), emphasizing modularity, clarity, and systematic development.

## Project Structure

```
.
├── flowsense/              # Main application package
│   ├── backend/            # Core processing logic
│   │   ├── api/
│   │   │   └── main.py     # FastAPI application
│   │   ├── models/
│   │   │   ├── classifier.py # Model training script
│   │   │   └── srtft_model.pkl # Saved trained model
│   │   ├── eeg_preprocessing.py
│   │   └── feature_extraction.py
│   ├── data/               # Data storage & generation scripts
│   │   ├── raw/            # Raw data (e.g., synthetic_eeg_data.csv)
│   │   ├── processed/      # (Placeholder)
│   │   └── synthetic/
│   │       └── generate_eeg.py
│   ├── frontend/           # Streamlit frontend application
│   │   └── app.py
│   ├── tests/              # Unit tests
│   │   └── test_srtft_pipeline.py
│   ├── docs/               # (Placeholder for detailed documentation)
│   │   ├── architecture.md
│   │   └── user_manual.md
│   └── run.py              # Script for console-based real-time simulation
│
├── .dockerignore           # Specifies files to ignore for Docker builds
├── Dockerfile              # Defines the Docker image for the web application
├── project_manifest.yaml   # Project metadata
├── README.md               # This file
├── requirements.txt        # Python package dependencies
└── run_web_srtft.sh        # Shell script to launch web API and UI
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
    Install all required packages using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include `numpy`, `pandas`, `scikit-learn`, `joblib`, `scipy`, `fastapi`, `uvicorn`, `pydantic`, `streamlit`, `requests`.

## Usage

### Console Simulation Usage

These commands allow you to generate data, train the model, and run a console-based simulation without web components. Ensure you are in the project root directory.

1.  **Generate Synthetic EEG Data**:
    ```bash
    python flowsense/data/synthetic/generate_eeg.py
    ```
    This creates `flowsense/data/raw/synthetic_eeg_data.csv`.

2.  **Train the Model**:
    ```bash
    python flowsense/backend/models/classifier.py
    ```
    This processes data, trains a model, and saves it as `flowsense/backend/models/srtft_model.pkl`.

3.  **Run the Real-Time Console Simulation**:
    ```bash
    python flowsense/run.py
    ```
    This loads the model and simulates real-time focus detection to the console. Press `Ctrl+C` to stop.

4.  **Run Unit Tests**:
    ```bash
    python -m unittest flowsense/tests/test_srtft_pipeline.py
    ```

### Web Application Usage (API & UI)

This mode provides a web-based interface to the SRTFT.

1.  **Ensure Dependencies**: If not using Docker (see below), make sure all dependencies from `requirements.txt` are installed.
2.  **Run the Web Application**:
    Execute the helper script from the project root:
    ```bash
    ./run_web_srtft.sh
    ```
    This script will:
    - Start the FastAPI backend API, typically on `http://localhost:8000`.
    - Start the Streamlit frontend UI, typically on `http://localhost:8501`.
    Access the UI by opening `http://localhost:8501` in your web browser. Press `Ctrl+C` in the terminal running the script to stop both servers.

### Docker Usage

Containerizing the application with Docker provides a consistent and isolated environment for the web application.

1.  **Build the Docker Image**:
    From the project root directory (where the `Dockerfile` is located):
    ```bash
    docker build -t srtft-app .
    ```
    This will create a Docker image named `srtft-app`.

2.  **Run the Docker Container**:
    ```bash
    docker run -d -p 8000:8000 -p 8501:8501 --name srtft-container srtft-app
    ```
    - `-d`: Runs the container in detached mode.
    - `-p 8000:8000`: Maps port 8000 of the container (FastAPI) to port 8000 on your host.
    - `-p 8501:8501`: Maps port 8501 of the container (Streamlit) to port 8501 on your host.
    - `--name srtft-container`: Assigns a name to the running container for easier management.

    You can then access the Streamlit UI at `http://localhost:8501` and the API at `http://localhost:8000`.

    To view logs from the container:
    ```bash
    docker logs srtft-container
    ```

    To stop the container:
    ```bash
    docker stop srtft-container
    ```
    To remove the container (after stopping):
    ```bash
    docker rm srtft-container
    ```

## Dependencies

This project relies on Python 3.9+ and the following key Python packages (see `requirements.txt` for a complete list):

-   **Core ML & Data Handling**:
    -   `numpy`
    -   `pandas`
    -   `scikit-learn`
    -   `joblib`
    -   `scipy`
-   **Web API & UI**:
    -   `fastapi`
    -   `uvicorn` (with `standard` extras for performance)
    -   `pydantic`
    -   `streamlit`
    -   `requests` (for Streamlit to communicate with API)

This project was autonomously generated and managed by FlowSense AI (simulated by Jules).
