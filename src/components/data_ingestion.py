import os
from pathlib import Path
import pandas as pd 
from pymongo.mongo_client import MongoClient
from src.entity.config_entity import DataIngestionConfig
from src.logging import logger
from src.constants import *



class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        """
        Initializes the DataIngestion class with the provided configuration.

        Parameters:
        - config (DataIngestionConfig): The configuration for data ingestion.

        Returns:
        - None
        """
        self.config = config
        
    def export_collection_as_df(self,collection_name, db_name) -> pd.DataFrame:
        """
        Exports data from a MongoDB collection to a pandas DataFrame.

        Parameters:
        - collection_name (str): The name of the MongoDB collection.
        - db_name (str): The name of the MongoDB database.

        Returns:
        - pd.DataFrame: A pandas DataFrame containing the data from the MongoDB collection.
        """
        
        try:   
            client = MongoClient(MONGO_DB_URL)
            collection = client[db_name][collection_name]
            
            # Retrieve the data from the MongoDB collection
            cursor = collection.find()

            # Convert the MongoDB cursor to a list of dictionaries
            data_list = list(cursor)

            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(data_list)

            # drop "_id"
            if '_id' in df.columns:
                df =  df.drop('_id', axis=1)
            return df
        except Exception as e:
            raise e
    
    def export_data_into_feature_store_file_path(self):
        """
        Exports data from MongoDB collection and saves it into a feature store file.

        Returns:
        - str: The file path where the data is saved.
        """
        try:
            logger.info(f"Exporting data from mongodb")
            raw_file_path  = self.config.raw_data_dir
            os.makedirs(raw_file_path,exist_ok=True)

            taxi_data = self.export_collection_as_df(
                                                   collection_name= MONGO_COLLECTION_NAME,
                                                   db_name = MONGO_DATABASE_NAME
                                                        )
            
            
            logger.info(f"Saving exported data into feature store file path: {raw_file_path}")
        
            feature_store_file_path = os.path.join(raw_file_path,'nyc_taxi_data.csv')
            # Correct the above line to create the directory as well
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            
            taxi_data.to_csv(feature_store_file_path,index=False, index_label=False)
           

            return feature_store_file_path
            

        except Exception as e:
            raise e           
