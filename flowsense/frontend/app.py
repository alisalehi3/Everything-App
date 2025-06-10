import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import sys

# Adjust PYTHONPATH for potential direct runs or if modules are structured locally
# Assumes this script is in flowsense/frontend/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration ---
BACKEND_API_URL = "http://127.0.0.1:8000/predict_focus_window/" # Ensure this matches your FastAPI port
SYNTHETIC_DATA_PATH = os.path.join(project_root, "flowsense/data/raw/synthetic_eeg_data.csv")
SAMPLING_RATE = 250  # Hz, must match data generation and model training
WINDOW_SIZE_SECONDS = 2.0
SAMPLES_PER_WINDOW = int(WINDOW_SIZE_SECONDS * SAMPLING_RATE)
NUM_CHANNELS = 4 # Must match data generation
CHANNEL_COLUMNS = [f'ch{i+1}' for i in range(NUM_CHANNELS)]

# --- Helper Functions ---
@st.cache_data # Cache the data loading
def load_synthetic_data(data_path):
    if not os.path.exists(data_path):
        st.error(f"Synthetic data file not found: {data_path}. Please generate it first.")
        return None
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading synthetic data: {e}")
        return None

def get_next_data_window(df, current_index):
    """Extracts the next window of data from the DataFrame."""
    if df is None or current_index + SAMPLES_PER_WINDOW > len(df):
        return None, current_index # No more data or df is None

    window_df = df.iloc[current_index : current_index + SAMPLES_PER_WINDOW]

    # Prepare data in the format expected by the API
    # List of lists: [[ch1_samples], [ch2_samples], ..., [chN_samples]]
    eeg_data_list_of_lists = []
    for ch_col in CHANNEL_COLUMNS:
        if ch_col not in window_df.columns:
            st.error(f"Channel column {ch_col} not found in synthetic data.")
            return None, current_index
        eeg_data_list_of_lists.append(window_df[ch_col].tolist())

    next_index = current_index + SAMPLES_PER_WINDOW # Non-overlapping for simplicity here
    return eeg_data_list_of_lists, next_index

# --- Streamlit UI ---
st.set_page_config(page_title="SRTFT - FlowSense Demo", layout="centered")
st.title("🧠 FlowSense - Simple Real-Time Focus Tracker")
st.markdown("This interface simulates real-time focus state prediction by sending windows of synthetic EEG data to a backend API.")

# Load data
synthetic_df = load_synthetic_data(SYNTHETIC_DATA_PATH)

if synthetic_df is not None:
    st.sidebar.success(f"Synthetic EEG data loaded ({len(synthetic_df)} samples).")

    # Initialize session state for current data index
    if 'current_data_index' not in st.session_state:
        st.session_state.current_data_index = 0

    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

    if 'error_message' not in st.session_state:
        st.session_state.error_message = None

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Get Current Focus State", type="primary", use_container_width=True):
            st.session_state.error_message = None # Clear previous error

            current_index = st.session_state.current_data_index
            eeg_window_data, next_index = get_next_data_window(synthetic_df, current_index)

            if eeg_window_data:
                st.session_state.current_data_index = next_index

                payload = {
                    "eeg_data": eeg_window_data,
                    "sampling_rate": SAMPLING_RATE
                }

                try:
                    response = requests.post(BACKEND_API_URL, json=payload, timeout=10) # Added timeout
                    response.raise_for_status() # Raises an exception for HTTP errors (4XX or 5XX)

                    prediction_data = response.json()
                    st.session_state.last_prediction = prediction_data

                except requests.exceptions.ConnectionError:
                    st.session_state.error_message = "Connection Error: Could not connect to the backend API. Is it running?"
                    st.session_state.last_prediction = None
                except requests.exceptions.HTTPError as e:
                    st.session_state.error_message = f"API Error: {e.response.status_code} - {e.response.text}"
                    st.session_state.last_prediction = None
                except requests.exceptions.RequestException as e:
                    st.session_state.error_message = f"Request Error: An unexpected error occurred ({e})."
                    st.session_state.last_prediction = None
            else:
                st.session_state.error_message = "End of synthetic data reached or data window is invalid."
                st.session_state.last_prediction = None # Clear previous prediction
                st.session_state.current_data_index = 0 # Reset for next run

    with col2:
        if st.session_state.error_message:
            st.error(st.session_state.error_message)

        if st.session_state.last_prediction:
            pred_data = st.session_state.last_prediction
            state = pred_data.get("predicted_state", "N/A")
            confidence = pred_data.get("confidence", 0.0)

            if state == "Focused":
                st.success(f"**Predicted State:** {state} (Confidence: {confidence:.2f})")
            elif state == "Unfocused":
                st.warning(f"**Predicted State:** {state} (Confidence: {confidence:.2f})")
            else:
                st.info(f"**Predicted State:** {state} (Confidence: {confidence:.2f})")

            # Display a snippet of the data that was sent (optional)
            # This is a bit tricky as eeg_window_data is not directly kept in session_state for display here
            # For now, we'll just show the prediction.
            # st.write("Data corresponding to this prediction (first few samples of first channel):")
            # st.json(eeg_window_data[0][:5]) # This would require passing eeg_window_data or storing it
        else:
            st.info("Click the button to get the latest focus state prediction.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Data Index:** {st.session_state.current_data_index} / {len(synthetic_df) if synthetic_df is not None else 'N/A'}")
    if st.sidebar.button("Reset Data Index"):
        st.session_state.current_data_index = 0
        st.session_state.last_prediction = None
        st.session_state.error_message = None
        st.rerun()

else:
    st.warning("Could not load synthetic EEG data. Please ensure `flowsense/data/raw/synthetic_eeg_data.csv` exists.")

st.markdown("---")
st.caption("Powered by FlowSense AI.")

# To run this Streamlit app (from the project root directory):
# PYTHONPATH=. streamlit run flowsense/frontend/app.py
if __name__ == "__main__":
    print("To run this Streamlit app, use the command:")
    print("streamlit run flowsense/frontend/app.py --server.runOnSave true")
