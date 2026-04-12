from src.utils import save_object, evaluate_model
from src.exception import CustomException
from src.logger import logger
import sys

try:
    models = {
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression()
    }

    report = evaluate_model(X_train, y_train, X_test, y_test, models)

    best_model_name = max(report, key=report.get)
    best_model = models[best_model_name]

    save_object("artifacts/model.pkl", best_model)

    logger.info(f"Best model: {best_model_name}")

except Exception as e:
    raise CustomException(e, sys)