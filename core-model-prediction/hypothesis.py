import nltk
import joblib
import textstat
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gemma2b_dependencies import Gemma2BDependencies


class BaseModelHypothesis:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

        self.analyzer = SentimentIntensityAnalyzer()
        self.lexicon_df = pd.read_csv(
            "https://storage.googleapis.com/ta-ai-detector/datasets/NRC-Emotion-Lexicon.csv")
        self.emotion_lexicon = self.process_emotion_lexicon()
        self.gemma2bdependencies = Gemma2BDependencies()

        self.features_normalized_text_length = []
        self.features_not_normalized = []

        self.scaler_normalized_text_length = joblib.load(
            "scalers/scaler-normalized-text-length.joblib")
        self.scaler_not_normalized = joblib.load(
            "scalers/scaler-not-normalized.joblib")

    def process_emotion_lexicon(self):
        emotion_lexicon = {}
        for _, row in self.lexicon_df.iterrows():
            if row["word"] not in emotion_lexicon:
                emotion_lexicon[row["word"]] = []
            emotion_lexicon[row["word"]].append(row["emotion"])
        return emotion_lexicon

    def calculate_normalized_text_length_features(self, text: str) -> np.ndarray:
        self.features_normalized_text_length = self.extract_pos_features(
            text)
        self.features_normalized_text_length = self.features_normalized_text_length + \
            self.calculate_emotion_proportions(text)
        self.features_normalized_text_length.append(
            self.measure_unique_word_ratio(text))

        return self.scaler_normalized_text_length.transform(np.array(self.features_normalized_text_length).astype(np.float32).reshape(1, -1))

    def calculate_not_normalized_features(self, text: str) -> np.ndarray:
        self.features_not_normalized.append(
            self.measure_sentiment_intensity(text))
        self.features_not_normalized = self.features_not_normalized + \
            self.measure_readability(text)
        self.features_not_normalized.append(
            self.gemma2bdependencies.calculate_perplexity(text))
        self.features_not_normalized.append(
            self.gemma2bdependencies.calculate_burstiness(text))

        return self.scaler_not_normalized.transform(np.array(self.features_not_normalized).astype(np.float32).reshape(1, -1))

    def extract_pos_features(self, text: str):
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        desired_tags = ["JJ", "VB", "RB", "PRP", "DT", "IN", "NN", "NNS"]
        pos_counts = defaultdict(int, {tag: 0 for tag in desired_tags})

        for _, pos in pos_tags:
            if pos in pos_counts:
                pos_counts[pos] += 1

        total_words = len(words)
        pos_ratios = [pos_counts[tag] / total_words for tag in desired_tags]

        return pos_ratios

    def measure_sentiment_intensity(self, text: str):
        sentiment = self.analyzer.polarity_scores(text)
        return sentiment["compound"]

    def measure_readability(self, text: str):
        gunning_fog = textstat.gunning_fog(text)
        smog_index = textstat.smog_index(text)
        dale_chall_score = textstat.dale_chall_readability_score(text)

        return [gunning_fog, smog_index, dale_chall_score]

    def calculate_emotion_proportions(self, text: str):
        tokens = nltk.word_tokenize(text)

        total_tokens = len(tokens)

        emotion_counts = {emotion: 0 for emotion in [
            "negative", "positive", "fear", "anger", "trust", "sadness", "disgust", "anticipation", "joy", "surprise"]}

        for token in tokens:
            if token in self.emotion_lexicon:
                for emotion in self.emotion_lexicon[token]:
                    emotion_counts[emotion] += 1

        proportions = {emotion: count / total_tokens for emotion,
                       count in emotion_counts.items()}

        return [
            proportions["negative"], proportions["positive"], proportions["fear"], proportions["anger"], proportions["trust"],
            proportions["sadness"], proportions["disgust"], proportions["anticipation"], proportions["joy"], proportions["surprise"]
        ]

    def measure_unique_word_ratio(self, text: str):
        tokens = nltk.word_tokenize(text)
        total_words = len(tokens)

        unique_words = len(Counter(tokens).keys())

        return (unique_words / total_words)
