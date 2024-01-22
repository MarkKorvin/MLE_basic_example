# Importing required libraries
import pandas as pd
import logging
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
# CONF_FILE = os.getenv('CONF_PATH')
CONF_FILE = './settings.json'
from utils import singleton, get_project_dir, configure_logging


# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])


@singleton
# Singleton class for generating Iris data set
class IrisDatasetGenerator():
    def __init__(self):
        self.df = None
        self.target = None
        self.train_df = None
        self.inference_df = None

    # Method to create the Iris data
    def create(self, save_path: os.path, is_labeled: bool = True):
        if is_labeled:
            self.save(self.train_df, save_path)
        else:
            self.save(self.inference_df, save_path)

    def read_dataset(self) -> pd.DataFrame:
        logger.info("Creating Iris dataset...")
        iris = load_iris()
        self.df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.target = iris.target
        self._splitting_dataset()

    def _splitting_dataset(self):
        logger.info("Splitting dataset...")
        X = self.df.values
        X = self.scaling(X)
        y = self.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=42)

        self.train_df = pd.concat([pd.DataFrame(X_train, columns=[
                                  'f1', 'f2', 'f3', 'f4']), pd.DataFrame({'label': y_train})], axis=1)
        self.inference_df = pd.DataFrame(
            X_test, columns=['f1', 'f2', 'f3', 'f4'])

    def scaling(self, df):
        scaler = StandardScaler()
        return scaler.fit_transform(df)

    # Method to save data
    def save(self, df: pd.DataFrame, out_path: os.path):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)


# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    gen = IrisDatasetGenerator()
    gen.read_dataset()
    gen.create(save_path=TRAIN_PATH)
    gen.create(save_path=INFERENCE_PATH, is_labeled=False)
    logger.info("Script completed successfully.")
