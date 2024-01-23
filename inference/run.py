"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import StandardScaler
import joblib


# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from utils import get_project_dir, configure_logging, Neural_clf

# Defines paths
DATA_DIR = get_project_dir(conf["general"]["data_dir"])
MODEL_DIR = get_project_dir(conf["general"]["models_dir"])
RESULTS_DIR = get_project_dir(conf["general"]["results_dir"])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--infer_file",
    help="Specify inference data file",
    default=conf["inference"]["inp_table_name"],
)
parser.add_argument("--out_path", help="Specify the path to the output table")


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for dirpath, dirnames, filenames in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(
                latest, conf["general"]["datetime_format"] + ".pt"
            ) < datetime.strptime(
                filename, conf["general"]["datetime_format"] + ".pt"
            ):
                latest = filename
    return os.path.join(MODEL_DIR, latest)
    # TODO try-catch missing model! if no model present, raise error and exit.


def get_model_by_path(path):
    """Loads and returns the specified model"""
    try:
        model = torch.load(path)
        logging.info(f"Path of the model: {path}")
        return model
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        logging.info(f"Test set size is: {df.shape[0]}")
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict the results and join it with the infer_data"""
    scaler = joblib.load(os.path.join(DATA_DIR, "trainig_scaler.gz"))
    infer_scaled = scaler.transform(infer_data)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Do the inference
        output = model(torch.tensor(infer_scaled, dtype=torch.float32))
        predictions = torch.argmax(output, dim=1)
    infer_data["results"] = predictions
    return infer_data


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf["general"]["datetime_format"]) + ".csv"
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f"Results saved to {path}")


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    tic = time.time()
    results = predict_results(model, infer_data)
    toc = time.time()
    store_results(results, args.out_path)
    logging.info(f"Inference done in {(toc - tic):.2f} seconds.")


if __name__ == "__main__":
    main()
