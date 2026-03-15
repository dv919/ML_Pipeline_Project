import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the project root to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation


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
    try:
        from src.components.data_ingestion import DataIngestion

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

        from src.components.model_trainer import ModelTrainer
        model_trainer = ModelTrainer()
        best_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Best Model R2 Score: {best_score}")
        print("Best Model R2 Score:", best_score)

    except Exception as e:
        logging.error("Pipeline failed")
        raise e