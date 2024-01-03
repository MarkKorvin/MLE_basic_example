import pandas as pd
import os
import sys
import pickle
import logging
import json
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../models'))

sys.path.append(os.path.dirname(ROOT_DIR))
with open("settings.json", "r") as file:
    conf = json.load(file)

def get_latest_model_path():
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, "%d.%m.%Y_%H.%M.pickle") < \
                        datetime.strptime(filename, "%d.%m.%Y_%H.%M.pickle"):
                latest = filename
    return os.path.join(MODEL_DIR, latest)

def get_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def get_inference_data(path):
    df = pd.read_csv(path)
    return df
    

if __name__ == "__main__":
    path = get_latest_model_path()
    print(path)
    model = get_model(path)
    df = get_inference_data(os.path.join(DATA_DIR, "xor_inference_data.csv"))
    res = model.predict(df)
    print(res)
    