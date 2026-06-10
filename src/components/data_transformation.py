import os
import sys
import pandas as pd
import numpy as np
import pickle

# Keeping your path logic
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging

class DataTransformation:
    def __init__(self):
        # Define the path for saving the preprocessor once
        self.preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

    def get_data_transformer_object(self, X_train):
        """Creates and returns the preprocessing object."""
        try:
            numerical_columns = X_train.select_dtypes(include=["int64", "float64"]).columns
            categorical_columns = X_train.select_dtypes(include=["object"]).columns

            logging.info(f"Numerical columns: {numerical_columns.tolist()}")
            logging.info(f"Categorical columns: {categorical_columns.tolist()}")

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False)) # Recommended if you use certain models
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # 1. REMOVE OUTLIERS (Only on Train data to prevent leakage)
            target_column_name = train_df.columns[-1] # Assuming price is the last column
            numerical_columns = train_df.select_dtypes(include=["int64", "float64"]).columns.drop(target_column_name)
            
            logging.info("Applying outlier removal on training data")
            train_df = self.remove_outliers_iqr(train_df, numerical_columns)

            # 2. SEPARATE TARGET FROM FEATURES
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # 3. GET TRANSFORMER
            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)

            # 4. APPLY TRANSFORMATION
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 5. CONCATENATE FEATURES AND TARGET
            # This makes it easier to pass one object to the model trainer
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # 6. SAVE PREPROCESSOR
            os.makedirs(os.path.dirname(self.preprocessor_obj_file_path), exist_ok=True)
            with open(self.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessing_obj, f)

            logging.info("Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers_iqr(self, df, numerical_cols):
        """Internal helper to clean data."""
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
        return df