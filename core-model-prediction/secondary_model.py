import joblib
import numpy as np
import pandas as pd
from typing import List


class SecondaryModel:
    def __init__(self):
        self.scaler = joblib.load("scalers/secondary_scaler.joblib")
        self.model = joblib.load("models/secondary_weights.joblib")
        self.secondary_model_features = [
            "machine_probability", "backspace_count_normalized", "typing_duration_normalized",
            "letter_discrepancy_normalized", "cosine_sim_gpt35", "cosine_sim_gpt4o"
        ]

    def preprocess_input(self, secondary_model_features: List[float]) -> pd.DataFrame:
        features_df = pd.DataFrame(
            [secondary_model_features], columns=self.secondary_model_features)
        features_df[self.secondary_model_features] = self.scaler.transform(
            features_df[self.secondary_model_features])
        return features_df

    def predict(self, secondary_model_features: List[float]) -> float:
        return self.model.predict_proba(self.preprocess_input(secondary_model_features))[:, -1][0]
