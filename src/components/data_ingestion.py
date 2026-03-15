import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the project root to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self):
        self.raw_data_path = "artifacts/raw.csv"
        self.train_data_path = "artifacts/train.csv"
        self.test_data_path = "artifacts/test.csv"

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method")

        try:
            df = pd.read_csv("data/bmw_global_sales_2018_2025.csv")

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            df.to_csv(self.raw_data_path, index=False)

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info("Data ingestion completed")

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    print("Train file path:", train_data)
    print("Test file path:", test_data)