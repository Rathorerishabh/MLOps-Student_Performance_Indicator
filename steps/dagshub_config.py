import mlflow
import dagshub

def initialize_dagshub():
    """Initialize DagsHub and MLflow configurations."""
    mlflow.set_tracking_uri("https://dagshub.com/Rathorerishabh/MLOps-Student_Performance_Indicator.mlflow")
    
    # Check if the experiment exists
    experiment_name = "3New_Experiment"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Failed to check or create experiment: {e}")

    dagshub.init(repo_owner='Rathorerishabh', repo_name='MLOps-Student_Performance_Indicator', mlflow=True)
