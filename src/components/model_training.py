import numpy as np 
import pandas as pd 
import os 
import json
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from src.entity.config_entity import ModelTrainerConfig
from src.utils.common import save_bin

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config = config
        self.model_info = {}
        
    def read_train_data(self):
        
        train_data = np.load(self.config.train_data_path)
        X_train = train_data[:,:-1]
        y_train = train_data[:,-1]

        return X_train, y_train
        
    def model_tuning(self):
        X_train, y_train = self.read_train_data()
        
        xgb_params = {
            'tree_method': 'gpu_hist',  
            'gpu_id': 0,  
        }
        xgb_clf = XGBClassifier(**xgb_params)
        grid_search = GridSearchCV(estimator=xgb_clf, param_grid=self.config.model_param_grid , scoring='f1', cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_


        self.model_info['best_params'] = grid_search.best_params_
        self.model_info['f1_score'] = grid_search.best_score_

        with open(os.path.join(self.config.root_dir, 'best_model_info.json'), 'w') as f:
            json.dump(self.model_info, f)

        save_bin(best_estimator, path=os.path.join(self.config.root_dir, 'model.joblib'))
