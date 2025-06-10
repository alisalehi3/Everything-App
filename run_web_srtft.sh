#!/bin/bash

echo "----------------------------------------------------"
echo "Starting FlowSense Simple Real-Time Focus Tracker"
echo "Backend API (FastAPI) & Frontend UI (Streamlit)"
echo "----------------------------------------------------"

# Ensure Python modules can be found from the project root
export PYTHONPATH=${PYTHONPATH}:.
# Also ensure local pip installations are in PATH
export PATH="/home/swebot/.local/bin:$PATH"


# Function to clean up background processes on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    # Kill all background jobs of this script (primarily the backend)
    if jobs -p | grep -q .; then
      kill $(jobs -p) > /dev/null 2>&1
    fi
    # Specifically kill backend if it's still around (trap might not get all cases)
    if [ ! -z "$BACKEND_PID" ] && ps -p $BACKEND_PID > /dev/null; then
        echo "Explicitly killing backend PID: $BACKEND_PID"
        kill $BACKEND_PID > /dev/null 2>&1
    fi
    echo "Servers shut down."
    exit 0
}

# Trap Ctrl+C (SIGINT) and script termination (SIGTERM) to run cleanup
trap cleanup SIGINT SIGTERM

# Start the FastAPI backend server in the background
echo ""
echo "Starting FastAPI backend server on port 8000..."
uvicorn flowsense.backend.api.main:app --host 0.0.0.0 --port 8000 --log-level info &
BACKEND_PID=$!
echo "FastAPI backend server started with PID: $BACKEND_PID"

# Wait a few seconds for the backend to initialize
echo "Waiting for backend to start (5s)..."
sleep 5

# Check if backend started (simple check, can be more sophisticated)
if ! ps -p $BACKEND_PID > /dev/null 2>&1; then
    echo "ERROR: FastAPI backend server (PID: $BACKEND_PID) failed to start."
    # Attempt to kill the script's process group if backend fails, to prevent lingering.
    kill 0
    exit 1
fi
echo "Backend appears to be running (PID: $BACKEND_PID)."

# Start the Streamlit frontend server in the foreground
echo ""
echo "Starting Streamlit frontend server on port 8501..."
echo "Access the UI at http://localhost:8501"
echo "Press Ctrl+C to stop both servers."
streamlit run flowsense/frontend/app.py --server.port 8501 --server.headless true --server.runOnSave false

# If Streamlit exits (e.g. user closes its window or stops it), cleanup will be called by the trap.
# If script is terminated, cleanup will also be called.

echo "SRTFT Web Application stopped."
