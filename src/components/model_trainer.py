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
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False)
            }
            
            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                 "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
                
            }
            logging.info("Genrating report")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            
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
            
            logging.info(f"found best modedel name {best_model_name} with score {r2_score(y_test,pred)}")
            return r2_score(y_test,pred)
            
                
            
            
        except Exception as e:
            raise CustomException(e,sys)