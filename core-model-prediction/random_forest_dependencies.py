from gemma2b_dependencies import Gemma2BDependencies
from collections import Counter


class RandomForestDependencies:
    def __init__(self):
        self.gemma2bdependencies = Gemma2BDependencies()

    def calculate_features(self, question: str, answer: str, probability: float, backspace_count: int, typing_duration: int, letter_click_counts: dict[str, int]):
        cosine_similarity = self.gemma2bdependencies.calculate_cosine_similarity(
            question, answer)
        backspace_count_normalized = backspace_count / len(answer)
        typing_duration_normalized = typing_duration / len(answer)
        letter_discrepancy = self.calculate_letter_discrepancy(
            answer, letter_click_counts)

        return [
            cosine_similarity, probability, backspace_count_normalized,
            typing_duration_normalized, letter_discrepancy
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
