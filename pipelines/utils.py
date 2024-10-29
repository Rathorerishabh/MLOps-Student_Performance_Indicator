import logging 
import pandas as pd

from src.data_cleaning import DataCleaning, DataPreProcessStrategy

def get_data_for_test():
    try:
        df=pd.read_csv("/mnt/c/Users/risha/Desktop/retail-prize-optimization-MLOps/data/Student_Performance.csv")
        df=df.sample(n=100)
        preprocess_startegy= DataPreProcessStrategy()
        data_cleaning=DataCleaning(df,preprocess_startegy)
        df=data_cleaning.handle_data()
        df.drop(["Performance Index"],axis=1,inplace =True)
        result=df.to_json(orient="split")
        return  result
    except Exception as e:
        logging.error(e)
        raise e


