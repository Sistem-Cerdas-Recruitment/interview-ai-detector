from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from hypothesis import BaseModelHypothesis
from secondary_model_dependencies import SecondaryModelDependencies
from secondary_model import SecondaryModel
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
    gpt35_answer: str
    gpt4o_answer: str


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
    gpt35_answer = data.gpt35_answer
    gpt4o_answer = data.gpt4o_answer

    # Data preparation for 1st model
    hypothesis = BaseModelHypothesis()
    additional_features = hypothesis.calculate_features_dataframe(answer)

    # 1st model prediction
    main_model = PredictMainModel()
    main_model_probability = main_model.predict(
        answer, additional_features)

    # Data preparation for 2nd model
    secondary_model_dependencies = SecondaryModelDependencies()
    secondary_model_features = secondary_model_dependencies.calculate_features(
        answer, main_model_probability, backspace_count,
        letter_click_counts, gpt4o_answer)

    # 2nd model prediction
    secondary_model = SecondaryModel()
    secondary_model_probability = secondary_model.predict(
        secondary_model_features)

    return {
        "predicted_class": "AI" if secondary_model_probability > 0.57 else "HUMAN",
        "main_model_probability": str(main_model_probability),
        "secondary_model_probability": str(secondary_model_probability),
        "confidence": get_confidence(main_model_probability, secondary_model_probability)
    }


def get_confidence(main_model_output: float, secondary_model_output: int):
    threshold = 0.54
    if (main_model_output >= 0.8 and secondary_model_output >= threshold) or (main_model_output <= 0.2 and secondary_model_output <= 1 - threshold):
        return 'High Confidence'
    elif (0.5 < main_model_output < 0.8 and secondary_model_output >= threshold) or (0.2 < main_model_output <= 0.5 and secondary_model_output < threshold):
        return 'Partially Confident'
    else:
        return 'Low Confidence'
