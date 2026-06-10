import sys
import os

# Add the project root to sys.path if running this file directly
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

class TrainingPipeline:

    def start_training(self):
        try:
            logging.info("Starting Training Pipeline")

            # Step 1: Data Ingestion
            # Expecting: (train_path, test_path)
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"Ingestion complete. Train path: {train_path}")

            # Step 2: Data Transformation
            # Note: We catch 3 outputs here because your DataTransformation returns 3
            transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
                train_path, test_path
            )
            logging.info("Transformation complete. Preprocessor saved.")

            # Step 3: Model Training
            # Passes the numpy arrays to the trainer
            trainer = ModelTrainer()
            best_score = trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training Pipeline Completed. Best R2 Score: {best_score}")

            return best_score

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        score = pipeline.start_training()
        print(f"\nSUCCESS! Final Model R2 Score: {score}")
    except Exception as e:
        print(f"Pipeline failed: {e}")