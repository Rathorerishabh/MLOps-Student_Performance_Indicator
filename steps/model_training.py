import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from steps.dagshub_config import initialize_dagshub  # Import centralized config

# Initialize DagsHub and MLflow once
initialize_dagshub()

@step(experiment_tracker="mlflow_tracker")
def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, config: ModelNameConfig) -> RegressorMixin:
    """Trains the model on the ingested dataset."""
    try:
        model = None
        # Start an MLflow run
        with mlflow.start_run(nested=True):
            if config.model_name == "LinearRegression":
                mlflow.sklearn.autolog()  # Enable autologging
                model = LinearRegressionModel()
                trained_model = model.train(X_train, y_train)  # Train the model
                mlflow.sklearn.log_model(trained_model, "linear_regression_model")
                return trained_model
            else:
                raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e
