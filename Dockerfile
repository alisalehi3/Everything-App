# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Ensure pip itself is up-to-date and add /root/.local/bin to PATH for pip user installs
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt && \
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc && \
    chmod +x ~/.bashrc

# Add /root/.local/bin to PATH for subsequent RUN, CMD, ENTRYPOINT
ENV PATH /root/.local/bin:$PATH

# Copy the rest of the application code into the container
# This includes the 'flowsense' directory, 'project_manifest.yaml', 'run_web_srtft.sh'
COPY . .
# If .dockerignore is set up correctly, this will only copy necessary files.

# Make the run script executable
RUN chmod +x ./run_web_srtft.sh

# Expose ports for FastAPI backend and Streamlit frontend
EXPOSE 8000
EXPOSE 8501

# Define the command to run the application
# This will execute the script that starts both Uvicorn and Streamlit
CMD ["./run_web_srtft.sh"]
