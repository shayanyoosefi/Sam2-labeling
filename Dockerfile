FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy application code
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Download SAM2 model checkpoint
RUN mkdir -p dataset_images labels && \
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O sam2_hiera_large.pt

EXPOSE 7263

CMD ["python3", "app.py"]