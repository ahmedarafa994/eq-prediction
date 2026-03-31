FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if existed, but we'll inline them here
RUN pip install --no-cache-dir \
    torchaudio \
    pytorch-lightning \
    wandb \
    pyyaml \
    hydra-core

# Copy the project
COPY ./conf /app/conf
COPY ./dataset_pipeline /app/dataset_pipeline
COPY ./training /app/training
COPY ./*.py /app/

# Set env proxy for weights and biases tracking
ENV WANDB_SILENT=true

CMD ["python", "training/train.py", "--config", "conf/config.yaml", "--data_dir", "/data"]
