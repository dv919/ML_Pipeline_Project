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
        Output: R2 score of the best model (using cross-validation)
        
        This method trains multiple regression models with regularization to prevent overfitting.
        It uses 5-fold cross-validation to evaluate model generalization.
        """
        try:
            logging.info("Splitting training and testing input data")
            
            # Split features and target
            # [:, :-1] takes all columns except the last one
            # [:, -1] takes only the last column (the Fuel_Price_Index)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info(f"Training set size: {X_train.shape}")
            logging.info(f"Test set size: {X_test.shape}")

            # Define the models we want to test with regularization to prevent overfitting
            models = {
                "Random Forest": RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1,
                    max_depth=15,
                    min_samples_leaf=5,
                    min_samples_split=10
                ),
                "Decision Tree": DecisionTreeRegressor(
                    random_state=42,
                    max_depth=10,
                    min_samples_leaf=5,
                    min_samples_split=10
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=42,
                    validation_fraction=0.1,
                    n_iter_no_change=10
                ),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(
                    random_state=42,
                    early_stopping_rounds=10,
                    eval_metric='rmse'
                ),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            # Hyperparameters for each model (Tuning with regularization focus)
            params = {
                "Decision Tree": {
                    'max_depth': [8, 10, 12, 15],
                    'min_samples_leaf': [3, 5, 7],
                    'min_samples_split': [8, 10, 12]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [12, 15, 18],
                    'min_samples_leaf': [4, 5, 6]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.85, 0.9],
                    'n_estimators': [100, 150, 200],
                    'max_depth': [4, 5, 6]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.05, 0.1],
                    'n_estimators': [100, 150],
                    'max_depth': [4, 5, 6],
                    'subsample': [0.8, 0.9]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 150]
                }
            }

            # Evaluate models using the helper function in utils.py (with CV and overfitting detection)
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, params=params
            )

            # To get the best model score from the report dictionary (CV score)
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the report dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Retrieve the actual model object
            best_model = models[best_model_name]

            # If the best model is below a certain threshold, it's not good enough
            if best_model_score < 0.6:
                raise CustomException("No best model found with a cross-validation R2 score greater than 0.6")

            # Logging and Printing results
            print("\n" + "*"*90 + "\n")
            print(f"🏆 BEST MODEL FOUND!")
            print(f"   Model Name: {best_model_name}")
            print(f"   Cross-Validation R² Score: {best_model_score:.4f}")
            print("\nModel Ranking:")
            for rank, (model_name, score) in enumerate(sorted(model_report.items(), key=lambda x: x[1], reverse=True), 1):
                print(f"   {rank}. {model_name}: {score:.4f}")
            print("\n" + "*"*90 + "\n")
            
            logging.info(f"Best model found based on cross-validation: {best_model_name}")
            logging.info(f"Best CV R² Score: {best_model_score:.4f}")

            # Save the winning model object to a pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final verification on test set
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            logging.info(f"Final test set R² Score: {r2_square:.4f}")
            
            return best_model_score  # Return CV score, not test score (more reliable)

        except Exception as e:
            raise CustomException(e, sys)