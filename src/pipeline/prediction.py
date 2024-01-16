import joblib 
import pandas as pd
from src.constants import *
from src.utils.feature_eng import generate_features

class PredictionPipeline:
    def __init__(self):
        try:
            self.preprocessor = joblib.load(Path(PREPROCESSOR_OBJ_PATH))
            self.model = joblib.load(Path(MODEL_OBJ_PATH))
        except Exception as e:
            raise RuntimeError(f"Error loading preprocessor or model: {e}")

    def model_prediction(self, input_data:pd.DataFrame):
        """
        Make predictions using the loaded preprocessor and model.

        Args:
            data (pd.DataFrame): Input data for prediction.

        Returns:
            tuple: A tuple containing the predicted labels and predicted probabilities.
        """
        try:
            features = generate_features(input_data=input_data)
            data_transformed = self.preprocessor.transform(features)
            prediction = self.model.predict(data_transformed)
            predict_prob = self.model.predict_proba(data_transformed)[:,-1]
            return prediction, predict_prob
        except Exception as e:
            raise RuntimeError(f"Error making predictions: {e}")
  
        