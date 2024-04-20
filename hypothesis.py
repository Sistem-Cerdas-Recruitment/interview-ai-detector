import nltk
import joblib
import textstat
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gemma2b import Gemma2BDependencies


class BaseModelHypothesis:
    def __init__(self, question: str, answer: str):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

        self.question = question
        self.answer = answer

        self.analyzer = SentimentIntensityAnalyzer()
        self.lexicon_df = pd.read_csv(
            "https://storage.googleapis.com/ta-ai-detector/datasets/NRC-Emotion-Lexicon.csv")
        self.emotion_lexicon = self.process_emotion_lexicon()
        self.gemma2bdependencies = Gemma2BDependencies(
            self.question, self.answer)

        self.features_normalized_text_length = []
        self.features_not_normalized = []

        self.scaler_normalized_text_length = joblib.load(
            "scaler-normalized-text-length.joblib")
        self.scaler_not_normalized = joblib.load(
            "scaler-not-normalized.joblib")

    def process_emotion_lexicon(self):
        emotion_lexicon = {}
        for _, row in self.lexicon_df.iterrows():
            if row["word"] not in emotion_lexicon:
                emotion_lexicon[row["word"]] = []
            emotion_lexicon[row["word"]].append(row["emotion"])
        return emotion_lexicon

    def calculate_normalized_text_length_features(self):
        self.features_normalized_text_length = self.extract_pos_features(
            self.answer)
        self.features_normalized_text_length = self.features_normalized_text_length + \
            self.calculate_emotion_proportions(self.answer)
        self.features_normalized_text_length.append(
            self.measure_unique_word_ratio(self.answer))

        return self.scaler_normalized_text_length.transform(np.array(self.features_normalized_text_length).astype(np.float32).reshape(1, -1))

    def calculate_not_normalized_features(self):
        self.features_not_normalized.append(
            self.measure_sentiment_intensity(self.answer))
        self.features_not_normalized = self.features_not_normalized + \
            self.measure_readability(self.answer)
        self.features_not_normalized.append(
            self.gemma2bdependencies.calculate_perplexity(self.answer))
        self.features_not_normalized.append(
            self.gemma2bdependencies.calculate_burstiness(self.answer))

        return self.scaler_not_normalized.transform(np.array(self.features_not_normalized).astype(np.float32).reshape(1, -1))

    def extract_pos_features(self):
        words = nltk.word_tokenize(self.answer)
        pos_tags = nltk.pos_tag(words)
        desired_tags = ["JJ", "VB", "RB", "PRP", "DT", "IN", "NN", "NNS"]
        pos_counts = defaultdict(int, {tag: 0 for tag in desired_tags})

        for _, pos in pos_tags:
            if pos in pos_counts:
                pos_counts[pos] += 1

        total_words = len(words)
        pos_ratios = [pos_counts[tag] / total_words for tag in desired_tags]

        return pos_ratios

    def measure_sentiment_intensity(self):
        sentiment = self.analyzer.polarity_scores(self.answer)
        return sentiment["compound"]

    def measure_readability(self):
        gunning_fog = textstat.gunning_fog(self.answer)
        smog_index = textstat.smog_index(self.answer)
        dale_chall_score = textstat.dale_chall_readability_score(self.answer)

        return [gunning_fog, smog_index, dale_chall_score]

    def calculate_emotion_proportions(self):
        tokens = nltk.word_tokenize(self.answer)

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

    def measure_unique_word_ratio(self):
        tokens = nltk.word_tokenize(self.answer)
        total_words = len(tokens)

        unique_words = len(Counter(tokens).keys())

        return (unique_words / total_words)
