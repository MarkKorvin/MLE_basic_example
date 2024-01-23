import os
import logging
import torch
import torch.nn as nn


def get_project_dir(sub_dir: str) -> str:
    """Return path to a project subdirectory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), sub_dir))


def configure_logging() -> None:
    """Configures logging"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class Neural_clf(nn.Module):
    def __init__(self):
        super(Neural_clf, self).__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=8)
        self.fc3 = nn.Linear(in_features=8, out_features=3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(x)
        return x
