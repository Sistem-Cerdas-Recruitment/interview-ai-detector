from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import cosine_similarity
from collections import Counter
import numpy as np
from device_manager import DeviceManager
from google.cloud import secretmanager


class Gemma2BDependencies:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Gemma2BDependencies, cls).__new__(cls)
            token = cls._instance.access_hf_token_secret()
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(
                "google/gemma-2b", token=token)
            cls._instance.model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2b", token=token)
            cls._instance.device = DeviceManager()
            cls._instance.model.to(cls._instance.device)
        return cls._instance

    def access_hf_token_secret(self):
        client = secretmanager.SecretManagerServiceClient()
        name = "projects/ta-2-sistem-cerdas/secrets/HF_TOKEN/versions/1"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode('UTF-8')

    def calculate_perplexity(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Calculate the model's output
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return perplexity.item()

    def calculate_burstiness(self, text: str):
        # Tokenize the text using GPT-2 tokenizer
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Count token frequencies
        frequency_counts = list(Counter(tokens).values())

        # Calculate variance and mean of frequencies
        variance = np.var(frequency_counts)
        mean = np.mean(frequency_counts)

        # Compute Variance-to-Mean Ratio (VMR) for burstiness
        vmr = variance / mean if mean > 0 else 0
        return vmr
