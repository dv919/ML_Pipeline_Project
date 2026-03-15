import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add the project root to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.logger import logging
from src.exception import CustomException # Fixed spelling from 'Custme'
from sklearn.metrics import r2_score # Use Regression metrics
from sklearn.model_selection import GridSearchCV

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
    Evaluates multiple regression models using GridSearchCV 
    and returns a report of R2 scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[model_name]

            logging.info(f"Evaluating model: {model_name}")

            # Hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            # Re-train model with best parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_test_pred = model.predict(X_test)

            # R2 Score is the standard for price prediction
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report # Return must be OUTSIDE the for-loop

    except Exception as e:
        raise CustomException(e, sys)