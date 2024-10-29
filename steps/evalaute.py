import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

try:
    experiment_tracker = Client().active_stack.experiment_tracker
except Exception as e:
    raise ValueError("Could not retrieve the active experiment tracker. Ensure you have an active ZenML stack.") from e

experiment_name = "Continuous_Deployment_Experiment"
mlflow.set_experiment(experiment_name)

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
                   ) -> Tuple[
                       Annotated[float, "mse"],
                       Annotated[float, "r2_score"],
                       Annotated[float, "rmse"],
                   ]:
    """Evaluate the model on the ingested dataset"""
    try:
        # Start MLflow run context
        experiment_name = "Mlflow_Experiment"
        mlflow.set_experiment(experiment_name)
    
    # Start a new MLflow run
        with mlflow.start_run(nested=True):
            prediction = model.predict(X_test)
            mse_class = MSE()
            mse = mse_class.calculate_scores(y_test, prediction)
            mlflow.log_metric("mse", mse)

            r2_class = R2()
            r2 = r2_class.calculate_scores(y_test, prediction)
            mlflow.log_metric("r2", r2)

            rmse_class = RMSE()
            rmse = rmse_class.calculate_scores(y_test, prediction)
            mlflow.log_metric("rmse", rmse)

    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise e

    return mse, r2, rmse
