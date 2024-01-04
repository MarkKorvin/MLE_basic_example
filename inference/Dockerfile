# Dockerfile for training
# Start from a base Python image
FROM python:3.10

ARG model_name="prod_model.pickle"

WORKDIR /app

# Copy data folder to Docker
COPY ../data/ /app/data

# Copy models folder to Docker
COPY ../models/${model_name} /app/models/${model_name}.pickle

# Copy the code itself
COPY inference /app/inference

# Copy utils.py
COPY ../utils.py /app
COPY ../settings.json /app

# Install any necessary packages listed in requirements.txt
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 inference/run.py