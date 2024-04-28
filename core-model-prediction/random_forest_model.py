import joblib
import numpy as np
from typing import List


class RandomForestModel:
    def __init__(self):
        self.scaler = joblib.load("scalers/rf_scaler.joblib")
        self.model = joblib.load("models/random_forest.joblib")

    def preprocess_input(self, secondary_model_features: List[float]) -> np.ndarray:
        return self.scaler.transform(np.array(secondary_model_features).astype(np.float32).reshape(1, -1))

    def predict(self, secondary_model_features: List[float]):
        return int(self.model.predict(self.preprocess_input(secondary_model_features))[0])
