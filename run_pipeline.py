from zenml.pipelines import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from pipelines.pipeline import data_pipeline
from steps.model_training import train_model
from steps.evalaute import  evaluate_model


data_path = "/mnt/c/Users/risha/Desktop/retail-prize-optimization-MLOps/data/Student_Performance.csv"

# Create and run the pipeline instance
if __name__ == "__main__":
    # Initialize the pipeline with step instances
    data_pipeline(ingest_df=ingest_df,
                  clean_df=clean_df,
                  train_model=train_model,
                  evaluate_model=evaluate_model)  
