import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    # Path where the best model will be saved
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Input: Transformed train and test arrays from DataTransformation
        Output: R2 score of the best model
        """
        try:
            logging.info("Splitting training and testing input data")
            
            # Split features and target
            # [:, :-1] takes all columns except the last one
            # [:, -1] takes only the last column (the price)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define the models we want to test
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameters for each model (Tuning)
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models using the helper function in utils.py
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, params=params
            )

            # To get the best model score from the report dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the report dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Retrieve the actual model object
            best_model = models[best_model_name]

            # If the best model is below a certain threshold, it's not good enough
            if best_model_score < 0.6:
                raise CustomException("No best model found with an R2 score greater than 0.6")

            # Logging and Printing results
            print("\n" + "*"*87 + "\n")
            print(f"Best Model Found! \nModel Name: {best_model_name} \nR2 Score: {best_model_score}")
            print("\n" + "*"*87 + "\n")
            
            logging.info(f"Best model found on both training and testing dataset: {best_model_name}")

            # Save the winning model object to a pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final prediction to verify
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)