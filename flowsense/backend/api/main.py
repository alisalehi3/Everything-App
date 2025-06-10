from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import numpy as np
import joblib
import os
import sys
from typing import List

# Adjust PYTHONPATH for sibling module imports
# Assumes this script is in flowsense/backend/api/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flowsense.backend.feature_extraction import extract_features
except ImportError as e:
    print(f"Error importing feature_extraction: {e}. Ensure PYTHONPATH is set correctly.")
    # Allow FastAPI to start, but endpoint will fail if this import fails.
    extract_features = None

# --- Configuration ---
MODEL_PATH = os.path.join(project_root, 'flowsense/backend/models/srtft_model.pkl')
# These should match the data generation and training
EXPECTED_NUM_CHANNELS = 4
WINDOW_SIZE_SECONDS = 2.0
# SAMPLING_RATE will be passed in request for now, but could be fixed if always constant

# --- Globals ---
app = FastAPI(title="FlowSense SRTFT API", version="1.0")
model = None

# --- Pydantic Models for Request/Response ---
class EEGWindowInput(BaseModel):
    eeg_data: List[conlist(float, min_length=1)] # List of lists (channels) of floats
    sampling_rate: int = 250 # Default to what we used, but allow override

    class Config:
        json_schema_extra = { # Renamed from schema_extra for Pydantic V2
            "example": {
                "eeg_data": [
                    [1.0, 2.0, ..., 0.5], # Channel 1 data (window_samples long)
                    [0.5, 1.5, ..., 0.3], # Channel 2 data
                    [2.0, 1.0, ..., 0.8], # Channel 3 data
                    [1.2, 0.8, ..., 0.6]  # Channel 4 data
                ],
                "sampling_rate": 250
            }
        }

class PredictionResponse(BaseModel):
    predicted_state: str
    confidence: float
    raw_prediction: int # 0 or 1

# --- API Events ---
@app.on_event("startup")
async def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        model = None # Or raise an error to prevent startup
        return
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FlowSense Simple Real-Time Focus Tracker API"}

@app.post("/predict_focus_window/", response_model=PredictionResponse)
async def predict_focus(eeg_input: EEGWindowInput):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or error during loading.")
    if extract_features is None:
        raise HTTPException(status_code=500, detail="Feature extraction module not available.")

    # Validate input data structure
    if len(eeg_input.eeg_data) != EXPECTED_NUM_CHANNELS:
        raise HTTPException(status_code=400,
                            detail=f"Invalid number of channels. Expected {EXPECTED_NUM_CHANNELS}, got {len(eeg_input.eeg_data)}.")

    # Calculate expected samples per window based on passed sampling_rate
    expected_samples_per_window = int(WINDOW_SIZE_SECONDS * eeg_input.sampling_rate)

    # Ensure all channels have the same number of samples and it matches expected
    samples_per_channel = [len(ch_data) for ch_data in eeg_input.eeg_data]
    if not all(s == samples_per_channel[0] for s in samples_per_channel):
        raise HTTPException(status_code=400, detail="All channels must have the same number of samples.")

    if samples_per_channel[0] != expected_samples_per_window:
        raise HTTPException(status_code=400,
                            detail=f"Invalid number of samples per channel. Expected {expected_samples_per_window} for {WINDOW_SIZE_SECONDS}s at {eeg_input.sampling_rate}Hz, got {samples_per_channel[0]}.")

    try:
        # Convert to NumPy array for feature extraction: (1, num_channels, window_samples)
        live_window_np = np.array(eeg_input.eeg_data).reshape(1, EXPECTED_NUM_CHANNELS, expected_samples_per_window)

        # Extract features
        current_features = extract_features(live_window_np, eeg_input.sampling_rate) # Should be (1, num_features)
        if current_features.size == 0:
            raise HTTPException(status_code=500, detail="Feature extraction failed or resulted in no features.")

        # Predict
        raw_pred = model.predict(current_features)[0]
        proba_pred = model.predict_proba(current_features)[0]

        state_str = "Focused" if raw_pred == 1 else "Unfocused"
        confidence_score = float(proba_pred[int(raw_pred)]) # Get probability of the predicted class

        return PredictionResponse(
            predicted_state=state_str,
            confidence=confidence_score,
            raw_prediction=int(raw_pred)
        )
    except ValueError as ve: # Catch potential errors from reshape or other numpy ops
        raise HTTPException(status_code=400, detail=f"Error processing input EEG data: {ve}")
    except Exception as e:
        # Log the exception details for server-side debugging
        print(f"Unhandled exception during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# To run this API (from the project root directory):
# PYTHONPATH=. uvicorn flowsense.backend.api.main:app --reload --port 8000
if __name__ == "__main__":
    # This is for direct execution if needed, but uvicorn is preferred for development
    print("To run this API, use: uvicorn flowsense.backend.api.main:app --reload --port 8000")
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
