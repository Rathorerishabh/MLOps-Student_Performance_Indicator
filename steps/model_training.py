import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

# Retrieve the active experiment tracker from ZenML
try:
    experiment_tracker = Client().active_stack.experiment_tracker
except Exception as e:
    raise ValueError("Could not retrieve the active experiment tracker. Ensure you have an active ZenML stack.") from e

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    """Trains the model on the ingested dataset."""
    try:
        model = None
        mlflow.set_experiment("Mlflow_Experiment")  # Set experiment name
        
        # Start an MLflow run
        with mlflow.start_run(nested=True):
            if config.model_name == "LinearRegression":
                mlflow.sklearn.autolog()  # Enable autologging
                
                model = LinearRegressionModel()
                trained_model = model.train(X_train, y_train)  # Train the model
                
                # Log the model explicitly (optional if using autolog)
                mlflow.sklearn.log_model(trained_model, "linear_regression_model")
                
                return trained_model
            else:
                raise ValueError("Model {} not supported".format(config.model_name))

    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e
