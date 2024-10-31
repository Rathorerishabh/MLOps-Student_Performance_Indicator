import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from steps.dagshub_config import initialize_dagshub  # Import centralized config

# Initialize DagsHub and MLflow once
initialize_dagshub()

@step(experiment_tracker="mlflow_tracker")
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[Annotated[float, "mse"], Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """Evaluate the model on the ingested dataset"""
    try:
        with mlflow.start_run(nested=True):
            prediction = model.predict(X_test)
            mse = MSE().calculate_scores(y_test, prediction)
            mlflow.log_metric("mse", mse)

            r2 = R2().calculate_scores(y_test, prediction)
            mlflow.log_metric("r2", r2)

            rmse = RMSE().calculate_scores(y_test, prediction)
            mlflow.log_metric("rmse", rmse)

    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise e

    return mse, r2, rmse
