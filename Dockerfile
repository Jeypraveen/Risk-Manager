# Pin to Bookworm explicitly so the base image doesn't silently drift
# to a newer Debian release (e.g. Trixie) that removes/renames packages.
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies required for OpenCV, PyTorch, and LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU first to avoid massive CUDA downloads
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Generate synthetic data and train the models so the image is ready to run
RUN python -m data.generate_data
RUN python -m src.tabular.train
RUN python -m scripts.train_meta_learner

# Expose the FastAPI default port
EXPOSE 8000

# Run the API server which also serves the static demo UI
CMD ["python", "src/api/server.py"]
