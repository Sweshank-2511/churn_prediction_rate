import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            prob = model.predict_proba(data_scaled)[:, 1]

            return preds, prob

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        session_time,
        clicks,
        pages_visited,
        last_login_days,
        purchase_count
    ):
        self.session_time = session_time
        self.clicks = clicks
        self.pages_visited = pages_visited
        self.last_login_days = last_login_days
        self.purchase_count = purchase_count

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "session_time": [self.session_time],
                "clicks": [self.clicks],
                "pages_visited": [self.pages_visited],
                "last_login_days": [self.last_login_days],
                "purchase_count": [self.purchase_count],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)