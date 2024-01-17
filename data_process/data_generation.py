# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json

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

# Singleton class for generating XOR data set
@singleton
class XorSetGenerator():
    def __init__(self):
        self.df = None

    # Method to create the XOR data
    def create(self, len: int, save_path: os.path, is_labeled: bool = True):
        logger.info("Creating XOR dataset...")
        self.df = self._generate_features(len)
        if is_labeled:
            self.df = self._generate_target(self.df)
        if save_path:
            self.save(self.df, save_path)
        return self.df

    # Method to generate features
    def _generate_features(self, n: int) -> pd.DataFrame:
        logger.info("Generating features...")
        x1 = np.random.choice([True, False], size=n)
        x2 = np.random.choice([True, False], size=n)
        return pd.DataFrame(list(zip(x1, x2)), columns=['x1', 'x2'])

    # Method to generate target
    def _generate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating target...")
        df['y'] = np.logical_xor(df['x1'], df['x2'])
        return df
    
    # Method to save data
    def save(self, df: pd.DataFrame, out_path: os.path):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    gen = XorSetGenerator()
    gen.create(len=256, save_path=TRAIN_PATH)
    gen.create(len=64, save_path=INFERENCE_PATH, is_labeled=False)
    logger.info("Script completed successfully.")