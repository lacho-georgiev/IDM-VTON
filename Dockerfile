# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Git Large File Storage (LFS)
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# Create a working directory
WORKDIR /app

# Clone the IDM-VTON repository
RUN git clone -b main https://github.com/lacho-georgiev/IDM-VTON

# Set up the virtual environment and install dependencies
RUN cd IDM-VTON && \
    python3.10 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install requests tqdm && \
    pip install -r requirements.txt && \
    pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade && \
    pip install xformers==0.0.24 && \
    pip install bitsandbytes==0.43.0 --upgrade && \
    pip install fastapi uvicorn

# Set the working directory to the cloned repo
WORKDIR /app/IDM-VTON

# Create a script to run uvicorn
RUN echo '#!/bin/bash\n. /app/IDM-VTON/venv/bin/activate\nexec uvicorn app:app --host 0.0.0.0 --port 7860' > /app/IDM-VTON/start_uvicorn.sh
RUN chmod +x /app/IDM-VTON/start_uvicorn.sh

# Expose port for FastAPI
EXPOSE 7860

# Command to run the application
CMD ["/app/IDM-VTON/start_uvicorn.sh"]
