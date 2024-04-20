from gemma2b import Gemma2BDependencies
from collections import Counter


class RandomForestDependencies:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

        self.gemma2bdependencies = Gemma2BDependencies(
            self.question, self.answer)
        self.random_forest_features = []

    def calculate_features(self, probability: float, backspace_count: int, typing_duration: int, letter_click_counts: dict[str, int]):
        cosine_similarity = self.gemma2bdependencies.calculate_cosine_similarity(
            self.question, self.answer)
        backspace_count_normalized = backspace_count / len(self.answer)
        typing_duration_normalized = typing_duration / len(self.answer)
        letter_discrepancy = self.calculate_letter_discrepancy(
            self.answer, letter_click_counts)

        self.random_forest_features = [
            cosine_similarity, probability, backspace_count_normalized,
            typing_duration_normalized, letter_discrepancy
        ]

    def calculate_letter_discrepancy(self, letter_click_counts: dict[str, int]):
        # Calculate letter frequencies in the text
        text_letter_counts = Counter(self.answer.lower())

        # Calculate the ratio of click counts to text counts for each letter, adjusting for letters not in text
        ratios = [letter_click_counts.get(letter, 0) / (text_letter_counts.get(letter, 0) + 1)
                  for letter in "abcdefghijklmnopqrstuvwxyz"]

        # Average the ratios and normalize by the length of the text
        average_ratio = sum(ratios) / len(ratios)
        discrepancy_ratio_normalized = average_ratio / \
            (len(self.answer) if len(self.answer) > 0 else 1)

        return discrepancy_ratio_normalized
