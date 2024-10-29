from zenml.pipelines import pipeline
from steps.ingest_data import ingest_df  # Ensure these are correctly imported
from steps.clean_data import clean_df
from steps.model_training import train_model
from steps.evalaute import evaluate_model

@pipeline(enable_cache=True)
def data_pipeline(ingest_df, clean_df, train_model,evaluate_model):
    # Ingest the data, passing the data path from the step
    ingest_df= ingest_df(data_path = "/mnt/c/Users/risha/Desktop/retail-prize-optimization-MLOps/data/Student_Performance.csv")
    
    # Clean the data
    X_train,X_test,y_train,y_test= clean_df(ingest_df)

    model = train_model(X_train,X_test,y_train,y_test)

    evaluate_model=evaluate_model(model,X_test,y_test)


