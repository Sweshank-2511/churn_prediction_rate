import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting training and test input data")

            #  Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logger.info("Data split completed")

            #  Define models
            models = {
                "LogisticRegression": LogisticRegression(),
                "RandomForest": RandomForestClassifier()
            }

            logger.info("Model training started")

            # Evaluate models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            logger.info(f"Model Report: {model_report}")

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            best_score = model_report[best_model_name]

            logger.info(f"Best Model Found: {best_model_name} with accuracy {best_score}")

            #  Optional check (good for interview)
            if best_score < 0.6:
                raise Exception("No good model found")

            #  Save best model
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            logger.info("Model saved successfully")

            #Final evaluation
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)