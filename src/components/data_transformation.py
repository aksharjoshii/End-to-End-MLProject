import os
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from src.logging import logger
from src.entity.config_entity import DataTransformationConfig
from src.constants import *
from src.utils.common import *
from src.utils.feature_eng import generate_features
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        
    def get_data_transformer_object(self):

        try:
            
            numerical_columns = ['passenger_count', 'mean_duration', 'mean_distance', 'predicted_fare',
                                'am_rush', 'day_time', 'pm_rush', 'night_time']
            categorical_columns = ['VendorID', 'RatecodeID', 'day', 
                                    'month', 'PULocationID', 'DOLocationID']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', categories='auto')),
                    ('scaler', StandardScaler(with_mean=False) )
                ]
            )

            preprocessor = ColumnTransformer(
                [

                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)

                ]
            )
            
            return preprocessor
        except Exception as e:
            raise e
    
    def data_splitting_preprocessing(self):
        
        data = pd.read_csv(self.config.data_path)
        data_clean =generate_features(data)
        data_clean.dropna(inplace=True)
        # split train and test data and save it as artifact 
        train, test = train_test_split(data_clean, test_size=0.25, random_state=0)
        train_x = train.drop([self.config.target_column], axis=1)
        train_y = train[[self.config.target_column]]
        test_x = test.drop([self.config.target_column], axis=1)
        test_y = test[[self.config.target_column]]
        
        preprocessor_obj = self.get_data_transformer_object()
        X_train_transformed = preprocessor_obj.fit_transform(train_x)
        X_test_transformed = preprocessor_obj.transform(test_x)
        
        # convert sparse matrix t0 numpy array  
        X_train_arr = X_train_transformed.toarray()
        X_test_arr = X_test_transformed.toarray()
        y_train_arr = train_y.to_numpy()
        y_test_arr = test_y.to_numpy()
        
        # save train  and test data as bumpy array. 
        train_arr = np.c_[X_train_arr, y_train_arr]
        test_arr = np.c_[X_test_arr,  y_test_arr]
        
        logger.info(f"save preprocessor object to {self.config.preprocessor_obj_file_path}")
        save_bin(object=preprocessor_obj, path =os.path.join(self.config.preprocessor_obj_file_path, "preprocessor.joblib"))
        
        
        # Save the NumPy arrays to .npy files
        np.save(os.path.join(self.config.root_dir, "train.npy"), train_arr)
        np.save(os.path.join(self.config.root_dir, "test.npy"), test_arr)
        logger.info("Preprocessed train and test data are saved ")
        logger.info(f"train data shape : {train_arr.shape}")
        logger.info(f"test data shape : {test_arr.shape}")
        
        