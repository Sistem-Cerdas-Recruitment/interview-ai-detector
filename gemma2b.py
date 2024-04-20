from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import cosine_similarity
from collections import Counter
import numpy as np


class Gemma2BDependencies:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
        self.device = torch.device("cuda")
        self.model.to(self.device)

    def calculate_perplexity(self):
        inputs = self.tokenizer(self.answer, return_tensors="pt",
                                truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Calculate the model's output
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return perplexity.item()

    def calculate_burstiness(self):
        # Tokenize the text using GPT-2 tokenizer
        tokens = self.tokenizer.tokenize(self.answer)

        # Count token frequencies
        frequency_counts = list(Counter(tokens).values())

        # Calculate variance and mean of frequencies
        variance = np.var(frequency_counts)
        mean = np.mean(frequency_counts)

        # Compute Variance-to-Mean Ratio (VMR) for burstiness
        vmr = variance / mean if mean > 0 else 0
        return vmr

    def get_embedding(self):
        inputs = self.tokenizer(self.text, return_tensors="pt",
                                truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_states = outputs.hidden_states[-1]
        # Average the token embeddings to get a sentence-level embedding
        embedding = torch.mean(last_hidden_states, dim=1)
        return embedding

    def calculate_cosine_similarity(self):
        embedding1 = self.get_embedding(self.question)
        embedding2 = self.get_embedding(self.answer)
        # Ensure the embeddings are in the correct shape for cosine_similarity
        return cosine_similarity(embedding1, embedding2).item()
