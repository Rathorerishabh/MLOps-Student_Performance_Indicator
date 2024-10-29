import logging
from abc import ABC,abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self,data:pd.DataFrame) ->Union[pd.DataFrame,pd.Series]:
        pass
class DataPreProcessStrategy(DataStrategy):
    """Strategy for preprocessing data"""
    
    def handle_data(self,data:pd.DataFrame) ->pd.DataFrame:
        """preprocess data"""
        try:
            data["Extracurricular Activities"]=data["Extracurricular Activities"].map({"Yes":1 ,"No":0})
            return data
        except Exception as e:
            logging.error(f"Error in DataPreProcessStrategy: {str(e)}")
            raise e
        
class DataDivideStrategy(DataStrategy):
    """Divide data into train and test sets"""
    def handle_data(self,data: pd.DataFrame)  ->Union[pd.DataFrame,pd.Series]:
        try:
            if isinstance(data, pd.Series):
            # Convert to DataFrame if necessary
              data = data.to_frame()
            print("Columns in DataFrame:", data.columns)  
            x=data.drop(["Performance Index"],axis=1)
            y=data["Performance Index"]
            X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error("Error in data divison:{}".format(e))
            raise e
        
class DataCleaning:
    """
    Class for cleanin data which process the data and divides it into train and test""" 
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy

    def handle_data(self) ->Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data:{}".format(e))
            raise e
if __name__=="__main__":
    data=pd.read_csv("/mnt/c/Users/risha/Desktop/retail-prize-optimization-MLOps/data/Student_Performance.csv")
    data_cleaning=DataCleaning(data,DataPreProcessStrategy)
    data_cleaning.handle_data()
