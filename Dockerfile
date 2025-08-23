FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend.py frontend.py ./
COPY models ./models

# Create folders for runtime
RUN mkdir -p dataset_images labels

# Expose ports
EXPOSE 5000 7263

# Healthcheck (waits for backend to be ready)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:5000/api/status || exit 1

# Run both backend and frontend
CMD ["bash", "-c", "python backend.py & python frontend.py"]
