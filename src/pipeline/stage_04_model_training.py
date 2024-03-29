from src.config.configuration import ConfigManager
from src.components.model_training import ModelTrainer
from src.logging import logger

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.model_tuning()

if __name__ == '__main__':
    
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

    
        