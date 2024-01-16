from src.config.configuration import ConfigManager
from src.components.model_evaluation import ModelEvaluation
from src.logging import logger

STAGE_NAME = 'Model Evaluation Stage'

class ModelEvaluationTrainingPipeline:
    
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
        model_eval_config = config.get_model_evaluation_config()
        model_eval_config = ModelEvaluation(config=model_eval_config)
        model_eval_config.save_eval_metrics()
    

if __name__ == '__main__':
    
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    