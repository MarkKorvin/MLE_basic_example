import pandas as pd
import os
import sys
import pickle
import logging
import json
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
with open("settings.json", "r") as file:
    conf = json.load(file)

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../models'))

TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])


RANDOM_STATE = 42


class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self):
        df = self.data_extraction(TRAIN_PATH)
        # ...
        return df

    def data_extraction(self, path):
        df = pd.read_csv(path)
        return df


class Training():
    def __init__(self) -> None:
        self.model = DecisionTreeClassifier(random_state=RANDOM_STATE)

    def run_training(self, df, out_path = None):
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=0.33)
        self.train(X_train, y_train)
        self.test(X_test, y_test)
        self.save(out_path)

    def data_split(self, df, test_size=0.33):
        X_train, X_test, y_train, y_test = train_test_split(df[['x1','x2']], 
                                                            df['y'], 
                                                            test_size=test_size,
                                                            random_state=RANDOM_STATE)
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        res = f1_score(y_test, y_pred)
        return res

    def save(self, path):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime("%d.%m.%Y_%H.%M") + ".pickle")
        else:
            path = os.path.join(MODEL_DIR, path)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


if __name__ == "__main__":
    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data()
    tr.run_training(df)
