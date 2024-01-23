# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

# Singleton class for generating Iris data set
@singleton
class IrisSetGenerator():
    def __init__(self):
        self.df = None

    # Method to create the Iris data
    def create(self, save_path_train: os.path, save_path_inference: os.path):
        logger.info("Creating Iris dataset...")
        self.df = self._load_iris_data()
        train_df, inference_df = self._split_data(self.df)
        if save_path_train:
            self.save(train_df, save_path_train)
        if save_path_inference:
            self.save(inference_df, save_path_inference)
        return train_df, inference_df

    # Method to load Iris data
    def _load_iris_data(self) -> pd.DataFrame:
        iris = datasets.load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        return df

    # Method to split data into training and inference sets
    def _split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Splitting data into training and inference sets...")
        train_df, inference_df = train_test_split(df, test_size=0.2, random_state=42)
        return train_df, inference_df
    
    # Method to save data
    def save(self, df: pd.DataFrame, out_path: os.path):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    gen = IrisSetGenerator()
    gen.create(save_path_train=TRAIN_PATH, save_path_inference=INFERENCE_PATH)
    logger.info("Script completed successfully.")