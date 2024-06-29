import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object





@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self) :
            self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            logging.info('Num_pipeline creating')
            num_pipe=Pipeline(steps=
                [('Imputer',SimpleImputer(strategy='median')),
                 ('SCaller',StandardScaler())
                    
                ]
            )
            logging.info('Catagorical_pipeline creating')
            
            cat_pipe=Pipeline(steps=[
                ('Imputer',SimpleImputer(strategy='most_frequent')),
                ('encoeder',OneHotEncoder()),
            ('SCaler',StandardScaler(with_mean=False))
            ])
            logging.info('Transformation_pipeline creating')
            
            preprocesser=ColumnTransformer([
                ('NUMBER',num_pipe,numerical_columns),
                ('CATAGORICAL',cat_pipe,categorical_columns)
            ])
            
            logging.info('Created object')
            
            return preprocesser
        
        
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info('Transformation has started')
            logging.info('Load data complete')
            
            logging.info('Getting preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column = 'math_score'
            numerical_columns = ["writing_score", "reading_score"]
            
            logging.info('Preparing training data')
            input_feature_train_df = train_data.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_data[target_column]
            
            logging.info('Preparing test data')
            input_feature_test_df = test_data.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_data[target_column]
            
            logging.info('Applying preprocessing on training data')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            logging.info('Applying preprocessing on test data')
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info('Ensuring artifacts directory exists')
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            
            logging.info(f'Saving preprocessor object to {self.data_transformation_config.preprocessor_obj_file_path}')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor object saved successfully')
            
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)