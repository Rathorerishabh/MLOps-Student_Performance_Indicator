import numpy as np
import json
import pandas as pd
from sklearn.metrics import accuracy_score
from .utils import get_data_for_test
from zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters
import mlflow

from steps.clean_data import clean_df
from steps.evalaute import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_training import train_model

# Docker settings to use MLFlow integration
docker_settings = DockerSettings(required_integrations=[MLFLOW])

# Define the Deployment Trigger configuration class
class DeploymentTriggerConfig(BaseParameters):
    """Deployment trigger config."""
    min_accuracy: float = 0.92

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(r2_score: float, config: DeploymentTriggerConfig):
    """Checks if the model accuracy meets the minimum threshold for deployment."""
    return r2_score > config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """Parameters for the MLFlow deployment loader step."""
    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool=True,
    model_name: str="model",
) -> MLFlowDeploymentService:
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLFlow deployment service found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name}. "
            f"Pipeline for the '{model_name}' model is currently running."
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        'Hours Studied',
        'Previous Scores',
        'Extracurricular Activities',
        'Sleep Hours',
        'Sample Question Papers Practiced'
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

# Define the pipeline using the decorator
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    # Start a new MLflow experiment
    
        # Ingest data
        df = ingest_df(data_path=data_path)

        # Clean and split the data
        X_train, X_test, y_train, y_test = clean_df(df)

        # Train the model
        model = train_model(X_train, X_test, y_train, y_test)

        # Evaluate the model
        mse, r2_score, rmse = evaluate_model(model, X_test, y_test)

        

        # Log parameters and metrics to MLflow

        # Trigger deployment based on the accuracy
        deployment_decision = deployment_trigger(r2_score)

        # Deploy the model
        mlflow_model_deployer_step(
            model=model,
            deploy_decision=deployment_decision,
            workers=1,
            timeout=DEFAULT_SERVICE_START_STOP_TIMEOUT
        )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str, model_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=service, data=data)
    return prediction
