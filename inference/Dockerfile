# Dockerfile for training
# Start from a base Python image
FROM python:3.10

ARG model_name=prod_model.pt
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy models folder to Docker
COPY models/${model_name} /app/models/${model_name}

# Copy the code itself
COPY inference /app/inference

# Copy usefull scripts and configs
COPY utils.py /app
COPY ${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
CMD ["python3", "inference/run.py"]