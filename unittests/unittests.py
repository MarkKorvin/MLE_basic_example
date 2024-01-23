import unittest
import pandas as pd
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = os.getenv("CONF_PATH")

from training.train import DataProcessor, Training


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = conf["general"]["data_dir"]
        cls.train_path = os.path.join(cls.data_dir, conf["train"]["table_name"])

    def test_data_extraction(self):
        dp = DataProcessor()
        df = dp.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_prepare_data(self):
        dp = DataProcessor()
        df = dp.prepare_data(100)
        self.assertEqual(df.shape[0], 100)


class TestTraining(unittest.TestCase):
    def test_train(self):
        tr = Training()
        # assume you have some prepared data
        X_train = pd.DataFrame({"x1": [1, 0, 1, 0], "x2": [1, 1, 0, 0]})
        y_train = pd.Series([0, 1, 1, 0])
        tr.train(X_train, y_train)
        self.assertIsNotNone(tr.model.tree_)


if __name__ == "__main__":
    unittest.main()
