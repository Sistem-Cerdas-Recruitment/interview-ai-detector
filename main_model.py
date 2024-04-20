from device_manager import DeviceManager
from transformers import AlbertModel, AlbertTokenizerFast
import torch.nn as nn
import torch
import numpy as np


class AlbertCustomClassificationHead(nn.Module):
    def __init__(self, albert_model, num_additional_features=25, dropout_rate=0.1):
        super(AlbertCustomClassificationHead, self).__init__()
        self.albert_model = albert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1024 + num_additional_features, 1)

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


class PredictMainModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictMainModel, cls).__new__()
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.model_name = "albert-large-v2"
        self.tokenizer = AlbertTokenizerFast.from_pretrained(self.model_name)
        self.albert_model = AlbertModel.from_pretrained(self.model_name)
        self.device = DeviceManager()

        self.model = AlbertCustomClassificationHead(
            self.albert_model).to(self.device)
        # TODO : CHANGE MODEL STATE DICT PATH
        self.model.load_state_dict(torch.load("models/albert_model.pth"))

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
