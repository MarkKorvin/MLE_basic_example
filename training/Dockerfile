# Dockerfile for training
# Start from a base Python image
FROM python:3.10

WORKDIR /app

# Copy data folder to Docker
COPY ../data/ /app/data

# Copy the code itself
COPY training /app/training

# Copy utils.py
COPY ../utils.py /app
COPY ../settings.json /app

# Install any necessary packages listed in requirements.txt
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 training/train.py