"""
Advanced Email Spam Classifier with Innovative Features
=======================================================

This module implements a comprehensive spam detection system with:
- Advanced text preprocessing
- Innovative feature engineering
- Multiple ML algorithms with ensemble methods
- Real-time classification capabilities
- Interactive visualizations
"""

import pandas as pd
import numpy as np
import re
import string
import warnings
from typing import List, Dict, Tuple, Optional
from collections import Counter
import pickle
import joblib

# Text processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class AdvancedTextPreprocessor:
    """Advanced text preprocessing with multiple techniques"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vader_analyzer = VaderAnalyzer()
        
        # Custom spam indicators
        self.spam_indicators = {
            'urgent_words': ['urgent', 'asap', 'immediately', 'now', 'today', 'limited time'],
            'money_words': ['free', 'cash', 'money', 'prize', 'win', 'winner', 'lottery', 'million'],
            'action_words': ['click', 'call', 'text', 'reply', 'claim', 'act now', 'buy now'],
            'suspicious_chars': ['!', '?', '$', '%', '&', '*', '+', '=', '@'],
            'phone_patterns': [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', r'\b\d{10,}\b'],
            'url_patterns': [r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'],
            'email_patterns': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b']
        }
    
    def clean_text(self, text: str) -> str:
        """Comprehensive text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', 'URL', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', 'EMAIL', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'PHONE', text)
        text = re.sub(r'\b\d{10,}\b', 'PHONE', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_linguistic_features(self, text: str) -> Dict:
        """Extract advanced linguistic features"""
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # Character analysis
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        features['whitespace_ratio'] = sum(1 for c in text if c.isspace()) / len(text) if text else 0
        
        # Spam indicator analysis
        features['urgent_word_count'] = sum(1 for word in self.spam_indicators['urgent_words'] if word in text)
        features['money_word_count'] = sum(1 for word in self.spam_indicators['money_words'] if word in text)
        features['action_word_count'] = sum(1 for word in self.spam_indicators['action_words'] if word in text)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_word_count'] = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
        
        # Sentiment analysis
        blob = TextBlob(text)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity
        
        vader_scores = self.vader_analyzer.polarity_scores(text)
        features['vader_compound'] = vader_scores['compound']
        features['vader_positive'] = vader_scores['pos']
        features['vader_negative'] = vader_scores['neg']
        features['vader_neutral'] = vader_scores['neu']
        
        # Readability features
        features['flesch_score'] = self._calculate_flesch_score(text)
        features['smog_score'] = self._calculate_smog_score(text)
        
        return features
    
    def _calculate_flesch_score(self, text: str) -> float:
        """Calculate Flesch Reading Ease Score"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))
    
    def _calculate_smog_score(self, text: str) -> float:
        """Calculate SMOG (Simple Measure of Gobbledygook) Score"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        
        if len(sentences) == 0:
            return 0
        
        score = 1.043 * np.sqrt(complex_words * (30 / len(sentences))) + 3.1291
        return score
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def preprocess_text(self, text: str) -> str:
        """Complete text preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)


class InnovativeFeatureExtractor:
    """Extract innovative features for spam detection"""
    
    def __init__(self):
        self.preprocessor = AdvancedTextPreprocessor()
    
    def extract_all_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract all features from text data"""
        features_list = []
        
        for text in texts:
            # Linguistic features
            linguistic_features = self.preprocessor.extract_linguistic_features(text)
            
            # Advanced spam detection features
            spam_features = self._extract_spam_specific_features(text)
            
            # Combine all features
            all_features = {**linguistic_features, **spam_features}
            features_list.append(all_features)
        
        return pd.DataFrame(features_list)
    
    def _extract_spam_specific_features(self, text: str) -> Dict:
        """Extract spam-specific features"""
        features = {}
        
        # Repetition analysis
        words = text.lower().split()
        word_freq = Counter(words)
        features['max_word_frequency'] = max(word_freq.values()) if word_freq else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        features['repeated_char_ratio'] = self._count_repeated_chars(text)
        
        # Structural features
        features['has_phone'] = 1 if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) else 0
        features['has_url'] = 1 if re.search(r'http[s]?://\S+', text) else 0
        features['has_email'] = 1 if re.search(r'\S+@\S+', text) else 0
        features['has_currency'] = 1 if re.search(r'[$£€¥₹]', text) else 0
        features['has_percentage'] = 1 if '%' in text else 0
        
        # Time-related features
        time_words = ['now', 'today', 'tomorrow', 'yesterday', 'week', 'month', 'year', 'hour', 'minute']
        features['time_word_count'] = sum(1 for word in time_words if word in text.lower())
        
        # Urgency features
        urgency_words = ['urgent', 'asap', 'immediately', 'hurry', 'quick', 'fast', 'instant']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text.lower())
        
        # Commercial features
        commercial_words = ['buy', 'sell', 'offer', 'deal', 'discount', 'sale', 'promotion', 'limited']
        features['commercial_score'] = sum(1 for word in commercial_words if word in text.lower())
        
        # Suspicious patterns
        features['consecutive_caps'] = len(re.findall(r'[A-Z]{3,}', text))
        features['consecutive_digits'] = len(re.findall(r'\d{4,}', text))
        features['consecutive_special'] = len(re.findall(r'[!?]{2,}', text))
        
        return features


class AdvancedSpamClassifier:
    """Advanced spam classifier with multiple algorithms and ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.feature_extractor = InnovativeFeatureExtractor()
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Clean and preprocess texts
        texts = df['v2'].fillna('').astype(str).tolist()
        labels = (df['v1'] == 'spam').astype(int).values
        
        # Extract features
        print("Extracting innovative features...")
        feature_df = self.feature_extractor.extract_all_features(texts)
        
        # Text preprocessing for vectorization
        preprocessed_texts = [self.feature_extractor.preprocessor.preprocess_text(text) for text in texts]
        
        # TF-IDF vectorization
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # Count vectorization
        self.vectorizers['count'] = CountVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        tfidf_features = self.vectorizers['tfidf'].fit_transform(preprocessed_texts)
        count_features = self.vectorizers['count'].fit_transform(preprocessed_texts)
        
        # Combine all features
        feature_array = feature_df.values
        combined_features = np.hstack([
            tfidf_features.toarray(),
            count_features.toarray(),
            feature_array
        ])
        
        return combined_features, labels, feature_df.columns.tolist()
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train multiple models"""
        print("Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_config = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(kernel='rbf', random_state=42, probability=True),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'catboost': CatBoostClassifier(random_state=42, verbose=False)
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train_balanced)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.4f}")
            
            self.models[name] = model
        
        # Create ensemble model
        print("Creating ensemble model...")
        ensemble_models = [
            ('nb', self.models['naive_bayes']),
            ('lr', self.models['logistic_regression']),
            ('rf', self.models['random_forest']),
            ('xgb', self.models['xgboost'])
        ]
        
        self.models['ensemble'] = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        self.models['ensemble'].fit(X_train_scaled, y_train_balanced)
        
        # Final evaluation
        y_pred_ensemble = self.models['ensemble'].predict(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        self.is_trained = True
        return X_test_scaled, y_test, y_pred_ensemble
    
    def predict(self, texts: List[str]) -> Dict:
        """Predict spam probability for given texts"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        feature_df = self.feature_extractor.extract_all_features(texts)
        preprocessed_texts = [self.feature_extractor.preprocessor.preprocess_text(text) for text in texts]
        
        # Vectorize
        tfidf_features = self.vectorizers['tfidf'].transform(preprocessed_texts)
        count_features = self.vectorizers['count'].transform(preprocessed_texts)
        
        # Combine features
        feature_array = feature_df.values
        combined_features = np.hstack([
            tfidf_features.toarray(),
            count_features.toarray(),
            feature_array
        ])
        
        # Scale and predict
        scaled_features = self.scaler.transform(combined_features)
        
        predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(scaled_features)[:, 1]
                predictions[name] = proba
            else:
                pred = model.predict(scaled_features)
                predictions[name] = pred
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.vectorizers = model_data['vectorizers']
        self.scaler = model_data['scaler']
        self.feature_extractor = model_data['feature_extractor']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")


def main():
    """Main function to run the spam classifier"""
    print("Advanced Email Spam Classifier")
    print("=" * 40)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].dropna()
    df.columns = ['label', 'text']
    
    print(f"Dataset shape: {df.shape}")
    print(f"Spam/Ham distribution:")
    print(df['label'].value_counts())
    
    # Initialize classifier
    classifier = AdvancedSpamClassifier()
    
    # Prepare data
    X, y, feature_names = classifier.prepare_data(df)
    
    # Train models
    X_test, y_test, y_pred = classifier.train_models(X, y)
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    classifier.save_model('spam_classifier_model.pkl')
    
    # Test with sample texts
    sample_texts = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        "Hey, how are you doing? Let's meet for coffee tomorrow.",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"
    ]
    
    print("\nSample Predictions:")
    predictions = classifier.predict(sample_texts)
    
    for i, text in enumerate(sample_texts):
        print(f"\nText {i+1}: {text[:50]}...")
        ensemble_prob = predictions['ensemble'][i]
        print(f"Spam Probability: {ensemble_prob:.4f}")
        print(f"Classification: {'SPAM' if ensemble_prob > 0.5 else 'HAM'}")


if __name__ == "__main__":
    main()
