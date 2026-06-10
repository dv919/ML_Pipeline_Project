import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Note: Paths should match where your Training Pipeline saves the files
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            # Transform raw input using the saved preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict the target (Fuel_Price_Index based on training data)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    This class is responsible for mapping the HTML/UI input to the
    format the model expects. (Matching the BMW sales dataset columns)
    """
    def __init__(self,
                 year: int,
                 month: int,
                 region: str,
                 model: str,
                 units_sold: int,
                 avg_price_eur: float,
                 revenue_eur: float,
                 bev_share: float,
                 premium_share: float,
                 gdp_growth: float):

        self.year = year
        self.month = month
        self.region = region
        self.model = model
        self.units_sold = units_sold
        self.avg_price_eur = avg_price_eur
        self.revenue_eur = revenue_eur
        self.bev_share = bev_share
        self.premium_share = premium_share
        self.gdp_growth = gdp_growth

    def get_data_as_data_frame(self):
        try:
            # Ensure keys match EXACTLY with the CSV column names used during training
            custom_data_input_dict = {
                "Year": [self.year],
                "Month": [self.month],
                "Region": [self.region],
                "Model": [self.model],
                "Units_Sold": [self.units_sold],
                "Avg_Price_EUR": [self.avg_price_eur],
                "Revenue_EUR": [self.revenue_eur],
                "BEV_Share": [self.bev_share],
                "Premium_Share": [self.premium_share],
                "GDP_Growth": [self.gdp_growth]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)