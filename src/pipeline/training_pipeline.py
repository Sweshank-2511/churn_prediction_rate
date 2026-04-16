import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logger


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logger.info("🚀 Training pipeline started")

            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            logger.info("Data ingestion completed")

            # Step 2: Data Transformation
            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(
                train_path, test_path
            )

            logger.info(" Data transformation completed")

            # Step 3: Model Training
            trainer = ModelTrainer()
            accuracy = trainer.initiate_model_trainer(train_arr, test_arr)

            logger.info(f"✅ Model training completed with accuracy: {accuracy}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)


#  MAIN EXECUTION BLOCK
if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        acc = pipeline.run_pipeline()

        print(f" Final Model Accuracy: {acc}")

    except Exception as e:
        print(e)