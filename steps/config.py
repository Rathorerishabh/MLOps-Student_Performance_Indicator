from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """ Model configs"""
    model_name: str ="LinearRegression" # List of models
    min_accuracy: float = 0.92
