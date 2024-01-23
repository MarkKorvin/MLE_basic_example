# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, "../data"))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining constants")
DATA_DIR = get_project_dir(conf["general"]["data_dir"])
TRAIN_PATH = os.path.join(DATA_DIR, conf["train"]["table_name"])
INFERENCE_PATH = os.path.join(DATA_DIR, conf["inference"]["inp_table_name"])
RAND_STATE = conf["general"]["random_state"]
TEST_SIZE = conf["train"]["test_size"]


def get_iris():
    # Load Iris dataset
    iris = load_iris()
    data = np.c_[iris.data, iris.target]
    columns = np.append(iris.feature_names, ["target"])
    iris_df = pd.DataFrame(data, columns=columns)

    # Split the dataset into features (X) and target variable (y)
    X = iris_df.drop("target", axis=1)
    y = iris_df["target"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RAND_STATE
    )

    # Save training set
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.to_csv(TRAIN_PATH, index=False)
    print(f"Training set saved to {TRAIN_PATH}")

    # Save test set without target: for later inference. this is not for evaluation. but "usage"
    infer_data = X_test
    infer_data.to_csv(INFERENCE_PATH, index=False)
    print(f"Data for inference saved to {INFERENCE_PATH}")


# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting data generation")
    get_iris()
    logger.info("Data generation completed successfully.")
