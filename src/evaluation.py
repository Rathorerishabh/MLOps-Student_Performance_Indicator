import logging 
from abc import ABC ,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class Evaluation(ABC):
    """Abstract class defining startegy for evaluation our models."""
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred: np.ndarray):
        """Calculate scores for the given predictions and true labels."""
        pass

class MSE(Evaluation):
    """ Calculate mean Squared error"""  
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating MSE") 
            mse=mean_squared_error(y_true,y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error calculating MSE: {}".format(e))
            raise e
        
class R2(Evaluation):
    """ Claculate r2 evaluation"""
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Claculating r2 squared scores")
            r2=r2_score(y_true,y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error calculating R2 score: {}".format(e))
            raise e
        

class RMSE(Evaluation):
    """ RMSE evaluation""" 
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse=np.sqrt(mean_squared_error(y_true,y_pred))
            logging.info("RMSE: {}".format(rmse))
            return rmse 
        except Exception as e:
            logging.error("Error calculating RMSE: {}".format(e))
            raise e