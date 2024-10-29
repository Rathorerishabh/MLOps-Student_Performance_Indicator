import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    The __init__ method is a constructor in Python classes. It's called when an instance (object) of the class is created
    The line self.data_path = data_path assigns the value of data_path to an instance variable self.data_path. 
    This allows the data_path to be accessed by other methods in the class.
    """
    def __init__(self, data_path:str):
        self.data_path=data_path

    def get_data(self):
        logging.info(f"Ingestion data from {self.data_path}")
        return pd.read_csv(self.data_path) 

@step
def ingest_df(data_path:str) ->pd.DataFrame:
    """
    When you put @step above a function, it tells a workflow management tool (like ZenML or Prefect) that this function is a "step" in a process or pipeline.
    Ingest data from a CSV file.
    Args:
    data_path (str): Path to the CSV file.
    Returns:
    pd.DataFrame: the ingested data
"""
    try:
        ingest_data = IngestData(data_path)
        df= ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error ingesting data: {e}")
        raise e