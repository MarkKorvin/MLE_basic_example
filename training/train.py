"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import numpy as np
import joblib

# Comment this lines if you have problems with MLFlow installation
import mlflow

mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv("CONF_PATH")

from utils import get_project_dir, configure_logging, Neural_clf

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf["general"]["data_dir"])
MODEL_DIR = get_project_dir(conf["general"]["models_dir"])
TRAIN_PATH = os.path.join(DATA_DIR, conf["train"]["table_name"])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_file",
    help="Specify training data file",
    default=conf["train"]["table_name"],
)
parser.add_argument("--model_path", help="Specify the path for the output model")


class DataProcessor:
    def __init__(self) -> None:
        pass

    def prepare_data(self) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)


class Training:
    def __init__(self) -> None:
        self.model = Neural_clf()

    def run_training(
        self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33
    ) -> None:
        logging.info("Running training")
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)
        X_train, X_test = self.scale_data(X_train, X_test)
        (
            X_train_tensors,
            y_train_tensors,
            X_test_tensors,
            y_test_tensors,
        ) = self.convert_to_tensors(X_train, y_train, X_test, y_test)
        train_dataset, test_dataset = self.prepare_dataloader(
            X_train_tensors,
            y_train_tensors,
            X_test_tensors,
            y_test_tensors,
            conf["train"]["batch_size"],
        )
        tic = time.time()
        self.model = self.train_model(
            train_dataloader=train_dataset,
            validation_dataloader=test_dataset,
            epochs=conf["train"]["epochs"],
        )
        toc = time.time()
        logging.info(f"Training done in {(toc - tic):.2f} seconds.")
        self.save(out_path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        logging.info("Train test split")
        X = df.loc[:, df.columns != "target"]
        y = df["target"]
        X_train, X_test, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=conf["general"]["random_state"]
        )
        logging.info(f"Training set size: {X_train.shape[0]}")
        logging.info(f"Validation set size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_val

    def convert_to_tensors(self, X_train, y_train, X_test, y_test):
        logging.info("Converting data to tensors")
        train_tensors = (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train.to_numpy(), dtype=torch.float32),
        )
        val_tensors = (
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test.to_numpy(), dtype=torch.float32),
        )
        return *train_tensors, *val_tensors

    def prepare_dataloader(
        self, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size
    ):
        logging.info("Creating training and test DataLoaders")
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    def scale_data(self, X_train, X_test) -> pd.DataFrame:
        logging.info("Scaling data")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, os.path.join(DATA_DIR, "trainig_scaler.gz"))
        return X_train, X_test

    def evaluate_model(self, validation_dataloader):
        # Evaluation on the test set
        self.model.eval()
        correct = 0
        criterion = nn.CrossEntropyLoss()
        val_losses = []
        with torch.no_grad():
            test_len = 0
            for data, targets in validation_dataloader:
                output = self.model(data).squeeze()
                # Convert probabilities to binary predictions
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == targets).sum().item()
                test_len += targets.shape[0]
                loss = criterion(output, targets.long())
                val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        accuracy = correct / test_len
        f1 = f1_score(targets, predictions, average="weighted")
        return (avg_val_loss, accuracy, f1)

    def train_model(self, train_dataloader, validation_dataloader, epochs):
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=conf["train"]["learning_rate"]
        )
        best_f1 = 0
        logging.info("Training started")
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            for data, targets in train_dataloader:
                optimizer.zero_grad()
                output = self.model(
                    data
                ).squeeze()  # Squeeze to remove any extra dimensions
                loss = criterion(output, targets.long())
                loss.backward()
                optimizer.step()
            validation_loss, accuracy, f1_sc = self.evaluate_model(
                validation_dataloader
            )
            if best_f1 < f1_sc:
                best_f1 = f1_sc
                best_model = deepcopy(self.model)
            logging.info(
                f"Epoch {epoch+1}, Avg validation loss: {validation_loss}, F1: {f1_sc}, acc: {accuracy}"
            )
        logging.info(f"Best F1: {best_f1}")
        return best_model

    def save(self, path: str) -> None:
        logging.info("Saving the model")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        if not path:
            path = os.path.join(
                MODEL_DIR,
                datetime.now().strftime(conf["general"]["datetime_format"]) + ".pt",
            )
        else:
            path = os.path.join(MODEL_DIR, path)
        torch.save(self.model, path)


def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data()
    tr.run_training(df, test_size=conf["train"]["test_size"])


if __name__ == "__main__":
    main()
