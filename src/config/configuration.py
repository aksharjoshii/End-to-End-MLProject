from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import *

class ConfigManager:
    
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config.artifacts_root])
    
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion 
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            db_uri= config.db_uri,
            raw_data_dir  = config.raw_data_dir    
        )
        return data_ingestion_config
    
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation 
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            data_path= config.data_path,
            STATUS_FILE  = config.STATUS_FILE,
            all_schema = schema  
        )
        return data_validation_config
    
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path= config.data_path,
            transformer_object_dir=config.transformer_object_dir
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModeltrainerConfig:
        config = self.config.model_trainer
        params = self.params.XGBoost
        schema = self.schema.TARGET
        
        create_directories([config.root_dir])
        
        model_trainer_config = ModeltrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            #TO_DO: params for xgboost
            target_column= schema.name    
        )
        
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.XGBoost
        schema = self.schema.TARGET
        
        create_directories([config.root_dir])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path = config.test_data_path,
            model_path = config.model_path,
            metric_file_name= config.metric_file_name,
            all_params = params,
            target_column= schema.name    
        )
        
        return model_evaluation_config
    
    
    