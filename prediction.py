from fastapi import FastAPI
from pydantic import BaseModel
from hypothesis import BaseModelHypothesis
from randomforest import RandomForestDependencies
import torch.nn as nn
import torch


class AlbertCustomClassificationHead(nn.Module):
    def __init__(self, albert_model, dropout_rate=0.1):
        super(AlbertCustomClassificationHead, self).__init__()
        self.albert_model = albert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1024 + 25, 1)

    def forward(self, input_ids, attention_mask, additional_features, labels=None):
        albert_output = self.albert_model(
            input_ids=input_ids, attention_mask=attention_mask).pooler_output

        combined_features = torch.cat(
            [albert_output, additional_features], dim=1)

        dropout_output = self.dropout(combined_features)

        logits = self.classifier(dropout_output)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            labels = labels.unsqueeze(1)
            loss = loss_fn(logits, labels.float())
            return logits, loss
        else:
            return logits


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

    return request_dict.get("backspace_count")
