import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import cosine_similarity
from collections import Counter
import numpy as np
from device_manager import DeviceManager


class Gemma2BDependencies:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Gemma2BDependencies, cls).__new__(cls)
            model_dir = os.getenv("MODEL_DIR", "/app/gemma-2b")
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            cls._instance.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
            cls._instance.device = DeviceManager()
            cls._instance.model.to(cls._instance.device)
        return cls._instance

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
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Count token frequencies
        frequency_counts = list(Counter(tokens).values())

        # Calculate variance and mean of frequencies
        variance = np.var(frequency_counts)
        mean = np.mean(frequency_counts)

        # Compute Variance-to-Mean Ratio (VMR) for burstiness
        vmr = variance / mean if mean > 0 else 0
        return vmr
