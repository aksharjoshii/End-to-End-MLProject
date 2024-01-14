import os
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from src.logging import logger
from src.entity.config_entity import DataTransformationConfig
from src.constants import *
from src.utils.common import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        
    def generate_features(self, input_data: pd.DataFrame) -> pd.DataFrame:

        # Convert pickup and dropoff cols to datetime
        input_data['tpep_pickup_datetime'] = pd.to_datetime(input_data['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
        input_data['tpep_dropoff_datetime'] = pd.to_datetime(input_data['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')
        #create month
        input_data['month'] = input_data['tpep_pickup_datetime'].dt.strftime('%b').str.lower()
        # create day col
        input_data['day'] = input_data['tpep_pickup_datetime'].dt.day_name().str.lower()
        # create time of the day
        input_data['am_rush'] = input_data['tpep_pickup_datetime'].dt.hour
        input_data['day_time'] = input_data['tpep_pickup_datetime'].dt.hour
        input_data['pm_rush'] = input_data['tpep_pickup_datetime'].dt.hour
        input_data['night time'] = input_data['tpep_pickup_datetime'].dt.hour

        input_data['am_rush'] = input_data['am_rush'].apply(lambda x: 1 if 6 <= x < 10 else 0)
        input_data['day_time'] = input_data['am_rush'].apply(lambda x: 1 if 10 <= x < 16 else 0)
        input_data['pm_rush'] = input_data['am_rush'].apply(lambda x: 1 if 16<= x < 20 else 0)
        input_data['night_time'] = input_data['am_rush'].apply(lambda x : 1 if (20 <= x < 24) or (0 <= x < 6) else 0)

        # drop redundant columns
        drop_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime',
                    'payment_type', 'trip_distance', 'store_and_fwd_flag',
                    'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
                    'improvement_surcharge', 'total_amount', 'tip_percent']
        # convert catergorical features to string
        cols_to_str = ['RatecodeID', 'VendorID', 'DOLocationID', 'PULocationID']

        # Convert each column to string
        for col in cols_to_str:
            input_data[col] = input_data[col].astype('str')

        input_data = input_data.drop(columns=drop_cols, axis=1)

        return input_data
        
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
        data_clean =self.generate_features(data)
        data_clean.dropna(inplace=True)
        column_names = data_clean.columns
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
        
        