import nltk
import joblib
import textstat
import pandas as pd
import numpy as np
from typing import List
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gemma2b_dependencies import Gemma2BDependencies
from string import punctuation


class BaseModelHypothesis:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')

        self.analyzer = SentimentIntensityAnalyzer()
        self.lexicon_df = pd.read_csv(
            "https://storage.googleapis.com/interview-ai-detector/higher-accuracy-final-model/NRC-Emotion-Lexicon.csv")
        self.emotion_lexicon = self.process_emotion_lexicon()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.gemma2bdependencies = Gemma2BDependencies()

        self.additional_feature_columns = [
            "nn_ratio", "nns_ratio", "jj_ratio", "in_ratio", "dt_ratio", "vb_ratio", "prp_ratio", "rb_ratio",
            "compound_score", "gunning_fog", "smog_index", "dale_chall_score",
            "negative_emotion_proportions", "positive_emotion_proportions", "fear_emotion_proportions",
            "anger_emotion_proportions", "trust_emotion_proportions", "sadness_emotion_proportions",
            "disgust_emotion_proportions", "anticipation_emotion_proportions", "joy_emotion_proportions",
            "surprise_emotion_proportions", "unique_words_ratio", "perplexity", "burstiness"
        ]

        self.features_normalized_text_length = [
            "nn_ratio", "nns_ratio", "jj_ratio", "in_ratio", "dt_ratio", "vb_ratio", "prp_ratio", "rb_ratio",
            "negative_emotion_proportions", "positive_emotion_proportions", "fear_emotion_proportions",
            "anger_emotion_proportions", "trust_emotion_proportions", "sadness_emotion_proportions",
            "disgust_emotion_proportions", "anticipation_emotion_proportions", "joy_emotion_proportions",
            "surprise_emotion_proportions", "unique_words_ratio"
        ]

        self.features_not_normalized = [
            "compound_score", "gunning_fog", "smog_index", "dale_chall_score",
            "perplexity", "burstiness"
        ]

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

    def calculate_features_dataframe(self, text: str) -> np.ndarray:
        normalized_text_length_features = self.calculate_normalized_text_length_features(
            text)
        not_normalized_features = self.calculate_not_normalized_features(text)
        all_features = normalized_text_length_features + not_normalized_features
        features_df = pd.DataFrame(
            [all_features], columns=self.additional_feature_columns)

        # Scaling features
        features_df[self.features_normalized_text_length] = self.scaler_normalized_text_length.transform(
            features_df[self.features_normalized_text_length])
        features_df[self.features_not_normalized] = self.scaler_not_normalized.transform(
            features_df[self.features_not_normalized])

        ordered_df = features_df[self.additional_feature_columns]

        return ordered_df.values.astype(np.float32).reshape(1, -1)

    def calculate_normalized_text_length_features(self, text: str) -> List[float]:
        pos_features = self.extract_pos_features(text)
        emotion_features = self.calculate_emotion_proportions(text)
        unique_word_ratio = [self.measure_unique_word_ratio(text)]
        features = pos_features + emotion_features + unique_word_ratio
        return features

    def calculate_not_normalized_features(self, text: str) -> List[float]:
        sentiment_intensity = [self.measure_sentiment_intensity(text)]
        readability_scores = self.measure_readability(text)
        perplexity = [self.gemma2bdependencies.calculate_perplexity(text)]
        burstiness = [self.gemma2bdependencies.calculate_burstiness(text)]
        features = sentiment_intensity + readability_scores + perplexity + burstiness
        return features

    def extract_pos_features(self, text: str):
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        desired_tags = ["NN", "NNS", "JJ", "IN", "DT", "VB", "PRP", "RB"]
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

    def __penn2morphy(self, penntag):
        morphy_tag = {
            'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n',  # Nouns
            'JJ': 'a', 'JJR': 'a', 'JJS': 'a',  # Adjectives
            'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',  # Verbs
            'RB': 'r', 'RBR': 'r', 'RBS': 'r',  # Adverbs
            # Pronouns, determiners, prepositions, modal verbs
            'PRP': 'n', 'PRP$': 'n', 'DT': 'n', 'IN': 'n', 'MD': 'v',
            # Others, treated as nouns unless a better fit is found
            'CC': 'n', 'CD': 'n', 'EX': 'n', 'FW': 'n', 'POS': 'n', 'TO': 'n', 'WDT': 'n', 'WP': 'n', 'WP$': 'n', 'WRB': 'n', 'PDT': 'n'
        }
        return morphy_tag.get(penntag[:2], 'n')

    def calculate_emotion_proportions(self, text: str):
        tokens = nltk.word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)

        lemmas = [self.lemmatizer.lemmatize(
            token.lower(), pos=self.__penn2morphy(tag)) for token, tag in tagged_tokens]

        total_lemmas = len(lemmas)

        emotion_counts = {emotion: 0 for emotion in [
            "negative", "positive", "fear", "anger", "trust", "sadness", "disgust", "anticipation", "joy", "surprise"]}

        for lemma in lemmas:
            if lemma in self.emotion_lexicon:
                for emotion in self.emotion_lexicon[lemma]:
                    emotion_counts[emotion] += 1

        proportions = {emotion: count / total_lemmas for emotion,
                       count in emotion_counts.items()}

        return [
            proportions["negative"], proportions["positive"], proportions["fear"], proportions["anger"], proportions["trust"],
            proportions["sadness"], proportions["disgust"], proportions["anticipation"], proportions["joy"], proportions["surprise"]
        ]

    def measure_unique_word_ratio(self, text: str):
        tokens = nltk.word_tokenize(text.lower())

        tokens = [token for token in tokens if token not in punctuation]

        total_words = len(tokens)

        unique_words = len(set(tokens))

        return (unique_words / total_words)
