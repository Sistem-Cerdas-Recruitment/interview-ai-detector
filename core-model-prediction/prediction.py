from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from hypothesis import BaseModelHypothesis
from random_forest_dependencies import RandomForestDependencies
from random_forest_model import RandomForestModel
from main_model import PredictMainModel
import numpy as np
from typing import List

app = FastAPI()


class PredictRequest(BaseModel):
    question: str
    answer: str
    backspace_count: int
    typing_duration: int
    letter_click_counts: dict[str, int]


class RequestModel(BaseModel):
    instances: List[PredictRequest]


@app.get("/health")
async def is_alive():
    return Response(status_code=status.HTTP_200_OK)


@app.post("/predict")
async def predict(request: RequestModel):
    responses = [process_instance(data) for data in request.instances]
    return {"predictions": responses}


def process_instance(data: PredictRequest):
    question = data.question
    answer = data.answer
    backspace_count = data.backspace_count
    typing_duration = data.typing_duration
    letter_click_counts = data.letter_click_counts

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
        "prediction_class": "AI" if secondary_model_prediction == 1 else "HUMAN",
        "details": {
            "main_model_probability": str(main_model_probability),
            "final_prediction": secondary_model_prediction
        }
    }
