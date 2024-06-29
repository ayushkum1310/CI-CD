import os
import sys
 
from src.logger import logging
from src.exception import CustomException 
from dataclasses import dataclass
 
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostClassifier,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRFRegressor
from src.utils import evaluate_models

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Spliting training and test')
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            
            
            models={
                "Random Jungle":RandomForestRegressor(),
                "Decision_tree":DecisionTreeRegressor(),
                "CAT":CatBoostRegressor(verbose=False)
            }
            logging.info("Genrating report")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            
            best_model=models[best_model_name]
            if best_model_score<0.60:
                raise CustomException("No model found")
            logging.info("Model found")
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            logging.info("Saved best model")
            pred=best_model.predict(X_test)
            
            
            return r2_score(y_test,pred)
            
                
            
            
        except Exception as e:
            raise CustomException(e,sys)