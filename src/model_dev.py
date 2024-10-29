import pandas as pd
import numpy as np
import logging 
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

class Model(ABC):
    @abstractmethod
    def train(self,X_train,y_train):
        pass

class LinearRegressionModel(Model):
    def train(self,X_train,y_train,**kwargs):
        try:
            reg=LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            return reg
        except Exception as e:
            logging.error("Error in training model:{}".format(e))
            raise e

#class Elasticnet(Model):
    #def train(self,X_train,y_train,**kwargs):
        #try:
            reg2=ElasticNet(**kwargs)
            reg2.fit(X_train,y_train)
            return reg2
        #except Exception as e:
            logging.error("Error in training model:{}".format(e))
            raise e

#class Ridge(Model):
    #def train(self,X_train,y_train,**kwargs):
        #try:
            reg3=Ridge(**kwargs)
            reg3.fit(X_train,y_train)
            return reg3
        #except Exception as e:
            logging.error("Error in training model:{}".format(e))
            raise e



    

