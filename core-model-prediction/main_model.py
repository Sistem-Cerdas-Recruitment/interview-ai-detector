from device_manager import DeviceManager
from transformers import AlbertModel, AlbertTokenizerFast
import torch.nn as nn
import torch
import numpy as np


class AlbertSeparateTransformation(nn.Module):
    def __init__(self, albert_model, num_additional_features=25,
                 hidden_size_albert=512, hidden_size_additional=128, classifier_hidden_size=256,
                 dropout_rate_albert=0.3, dropout_rate_additional=0.1, dropout_rate_classifier=0.1):
        super(AlbertSeparateTransformation, self).__init__()
        self.albert = albert_model

        # Transform ALBERT's features to an intermediate space
        self.albert_feature_transform = nn.Sequential(
            nn.Linear(1024, hidden_size_albert),
            nn.ReLU(),
            nn.Dropout(dropout_rate_albert),
        )

        # Transform additional features to an intermediate space
        self.additional_feature_transform = nn.Sequential(
            nn.Linear(num_additional_features, hidden_size_additional),
            nn.ReLU(),
            nn.Dropout(dropout_rate_additional),
        )

        # Combine both transformed features and process for final prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_albert + hidden_size_additional,
                      classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate_classifier),
            nn.Linear(classifier_hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, additional_features):
        albert_output = self.albert(
            input_ids=input_ids, attention_mask=attention_mask).pooler_output

        transformed_albert_features = self.albert_feature_transform(
            albert_output)
        transformed_additional_features = self.additional_feature_transform(
            additional_features)

        combined_features = torch.cat(
            (transformed_albert_features, transformed_additional_features), dim=1)

        logits = self.classifier(combined_features)
        return logits


class PredictMainModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictMainModel, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.model_name = "albert-large-v2"
        self.tokenizer = AlbertTokenizerFast.from_pretrained(self.model_name)
        self.albert_model = AlbertModel.from_pretrained(self.model_name)
        self.device = DeviceManager()

        self.model = AlbertSeparateTransformation(
            self.albert_model).to(self.device)
        self.model.load_state_dict(torch.load("models/albert_weights.pth"))

    def preprocess_input(self, text: str, additional_features: np.ndarray):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        additional_features_tensor = torch.tensor(
            additional_features, dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
            "additional_features": additional_features_tensor.to(self.device)
        }

    def predict(self, text: str, additional_features: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            data = self.preprocess_input(text, additional_features)
            logits = self.model(**data)
            return torch.sigmoid(logits).cpu().numpy()[0][0]
