import os 
from pathlib import Path
import numpy as np 
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, f1_score
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config
        
    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision  = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1  = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        
        return accuracy, precision, recall, f1, roc_auc
    def save_eval_metrics(self):
        test_data = np.load(self.config.test_data_path)
        X_test = test_data[:, :-1]
        y_test = test_data[:,-1]
        
        model = joblib.load(self.config.model_path)
        y_pred = model.predict(X_test)
        
        accuracy, precision, recall, f1, roc_auc = self.eval_metrics(actual=y_test, pred= y_pred)
        
        scores = {
                "accuracy_score": round(accuracy, 4),
                "precision_score": round(precision, 4),
                "recall_score": round(recall, 4),
                "f1_score": round(f1, 4),
                "roc_auc_score": round(roc_auc, 4)
                 }
        save_json(path=Path(self.config.metric_file_name), data= scores)
        