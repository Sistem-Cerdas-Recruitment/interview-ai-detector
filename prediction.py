from fastapi import FastAPI
from pydantic import BaseModel
from hypothesis import BaseModelHypothesis
from random_forest_dependencies import RandomForestDependencies
from random_forest_model import RandomForestModel
from main_model import PredictMainModel
import torch.nn as nn
import torch
import numpy as np

app = FastAPI()


class PredictRequest(BaseModel):
    question: str
    answer: str
    backspace_count: int
    typing_duration: int
    letter_click_counts: dict[str, int]


@app.post("/predict")
async def predict(request: PredictRequest):
    request_dict = request.model_dump()

    question = request_dict.get("question")
    answer = request_dict.get("answer")
    backspace_count = request_dict.get("backspace_count")
    typing_duration = request_dict.get("typing_duration")
    letter_click_counts = request_dict.get("letter_click_counts")

    hypothesis = BaseModelHypothesis()
    features_normalized_text_length = hypothesis.calculate_normalized_text_length_features(
        answer)
    features_not_normalized = hypothesis.calculate_not_normalized_features(
        answer)

    combined_additional_features = np.concatenate(
        (features_normalized_text_length, features_not_normalized), axis=1)

    main_model = PredictMainModel()
    main_model_probability = main_model.predict(
        answer, combined_additional_features)

    random_forest_features = RandomForestDependencies()
    secondary_model_features = random_forest_features.calculate_features(
        question, answer, main_model_probability, backspace_count, typing_duration, letter_click_counts)

    secondary_model = RandomForestModel()
    secondary_model_prediction = secondary_model.predict(
        secondary_model_features)

    return {
        "main_model_probability": main_model_probability,
        "final_prediction": secondary_model_prediction,
        "prediction_class": "AI" if secondary_model_prediction == 1 else "HUMAN"
    }
