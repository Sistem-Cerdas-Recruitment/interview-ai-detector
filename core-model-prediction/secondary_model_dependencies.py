from sentence_transformers import SentenceTransformer, util
from collections import Counter


class SecondaryModelDependencies:
    def __init__(self):
        self.text_similarity_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2')

    def calculate_features(self, answer: str, probability: float, backspace_count: int, typing_duration: int,
                           letter_click_counts: dict[str, int], gpt35_answer: str, gpt4o_answer: str):
        backspace_count_normalized = backspace_count / len(answer)
        typing_duration_normalized = typing_duration / len(answer)
        letter_discrepancy = self.calculate_letter_discrepancy(
            answer, letter_click_counts)

        cosine_sim_gpt35 = self.calculate_similarity_gpt35(
            answer, gpt35_answer)
        cosine_sim_gpt4o = self.calculate_similarity_gpt4o(
            answer, gpt4o_answer)

        return [
            probability, backspace_count_normalized, typing_duration_normalized,
            letter_discrepancy, cosine_sim_gpt35, cosine_sim_gpt4o
        ]

    def calculate_letter_discrepancy(self, text: str, letter_click_counts: dict[str, int]):
        # Calculate letter frequencies in the text
        text_letter_counts = Counter(text.lower())

        # Calculate the ratio of click counts to text counts for each letter, adjusting for letters not in text
        ratios = [letter_click_counts.get(letter, 0) / (text_letter_counts.get(letter, 0) + 1)
                  for letter in "abcdefghijklmnopqrstuvwxyz"]

        # Average the ratios and normalize by the length of the text
        average_ratio = sum(ratios) / len(ratios)
        discrepancy_ratio_normalized = average_ratio / \
            (len(text) if len(text) > 0 else 1)

        return discrepancy_ratio_normalized

    def calculate_similarity_gpt35(self, answer: str, gpt35_answer: str) -> float:
        embedding1 = self.text_similarity_model.encode(
            [answer], convert_to_tensor=True)
        embedding2 = self.text_similarity_model.encode(
            [gpt35_answer], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embedding1, embedding2)
        return cosine_scores.item()

    def calculate_similarity_gpt4o(self, answer: str, gpt4o_answer: str) -> float:
        embedding1 = self.text_similarity_model.encode(
            [answer], convert_to_tensor=True)
        embedding2 = self.text_similarity_model.encode(
            [gpt4o_answer], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embedding1, embedding2)
        return cosine_scores.item()
