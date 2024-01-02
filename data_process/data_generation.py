import numpy as np
import pandas as pd
import logging
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import singleton

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

TRAIN_PATH = os.path.join(DATA_DIR, 'xor_train_data.csv')
INFERENCE_PATH = os.path.join(DATA_DIR, 'xor_inference_data.csv')


@singleton
class XorSetGenerator():
    def __init__(self):
        self.df = None
    
    def create(self, len: int, save_path: os.path, is_labeled: bool = True):
        self.df = self._generate_features(len)
        if is_labeled:
            self.df = self._generate_target(self.df)
        if save_path:
            self.save(self.df, save_path)
        return self.df

    def _generate_features(self, n: int) -> pd.DataFrame:
        x1 = np.random.choice([True, False], size=n)
        x2 = np.random.choice([True, False], size=n) 
        return pd.DataFrame(list(zip(x1, x2)), columns = ['x1', 'x2']) 

    def _generate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df['y'] = np.logical_xor(df['x1'], df['x2']) 
        return df
    
    def save(self, df: pd.DataFrame, out_path: os.path):
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    gen = XorSetGenerator()
    gen.create(len=256, save_path=TRAIN_PATH)
    gen.create(len=64, save_path=INFERENCE_PATH, is_labeled=False)