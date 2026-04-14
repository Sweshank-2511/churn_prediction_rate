import os
import sys
import pickle

import numpy as np
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logger


#  Save object (model / preprocessor)
def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


#  Load object
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


#  Evaluate multiple models (IMPORTANT)
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            report[model_name] = accuracy

            logger.info(f"{model_name} accuracy: {accuracy}")

        return report

    except Exception as e:
        raise CustomException(e, sys)