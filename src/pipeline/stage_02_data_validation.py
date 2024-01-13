from src.config.configuration import ConfigManager
from src.logging import logger
from src.components.data_validation import DataValidation

STAGE_NAME = 'Data Validation Stage'

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
        data_valid_config = config.get_data_validation_config()
        data_valid = DataValidation(config=data_valid_config)
        data_valid.validate_data_columns()
    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    
    
    
    
    

   
        