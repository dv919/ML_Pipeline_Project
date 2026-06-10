import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add the project root to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates multiple regression models using GridSearchCV with cross-validation
    to detect overfitting. Returns a report of R2 scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[model_name]

            logging.info(f"Evaluating model: {model_name}")

            # Hyperparameter tuning with 5-fold cross-validation to prevent overfitting
            gs = GridSearchCV(model, para, cv=5, n_jobs=-1, verbose=0, scoring='r2')
            gs.fit(X_train, y_train)

            logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            logging.info(f"Best CV score: {gs.best_score_:.4f}")

            # Re-train model with best parameters on full training set
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions on test set
            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            # Calculate metrics on both train and test to detect overfitting
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Calculate cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Log metrics to detect overfitting (train R2 >> test R2 indicates overfitting)
            overfit_gap = train_r2 - test_r2
            logging.info(f"{model_name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Overfit Gap: {overfit_gap:.4f}")
            logging.info(f"{model_name} - CV Mean R²: {cv_mean:.4f} (+/- {cv_std:.4f}), Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

            # Use CV mean score for final report (more reliable than single test set)
            report[model_name] = cv_mean

        return report

    except Exception as e:
        raise CustomException(e, sys)