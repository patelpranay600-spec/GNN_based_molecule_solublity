FROM python:3.10-slim

# Install system dependencies required by RDKit and PyTorch
RUN apt-get update && apt-get install -y libxrender1 libxext6 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU first to avoid pulling massive GPU wheels
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code and the weights file
COPY main.py .
COPY schnet_weights.pt .

# Expose the port and run Uvicorn
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]