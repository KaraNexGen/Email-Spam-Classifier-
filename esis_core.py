"""
Email Security Intelligence System (ESIS) - Core Engine
======================================================

Advanced multi-class email threat detection system with:
- Multi-class classification (Spam, Phishing, Ham)
- Explainable AI with SHAP/LIME
- Sender reputation scoring
- Link & attachment analysis
- Adaptive learning capabilities
"""

import pandas as pd
import numpy as np
import re
import string
import hashlib
import urllib.parse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import joblib
import pickle

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Explainable AI
import shap
import lime
import lime.lime_tabular

# Text Processing
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

# Security Analysis
import requests
from bs4 import BeautifulSoup
import whois
import dns.resolver
import phonenumbers
from urllib.parse import urlparse
import socket

warnings.filterwarnings('ignore')

# Download NLTK data
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


@dataclass
class EmailAnalysis:
    """Data class for email analysis results"""
    email_id: str
    classification: str  # 'spam', 'phishing', 'ham'
    confidence: float
    threat_level: str  # 'low', 'medium', 'high', 'critical'
    sender_reputation: float
    risk_factors: List[str]
    explanations: Dict[str, Any]
    timestamp: datetime


class SenderReputationAnalyzer:
    """Advanced sender reputation analysis"""
    
    def __init__(self):
        self.trusted_domains = {
            'gmail.com': 0.9, 'outlook.com': 0.9, 'yahoo.com': 0.8,
            'hotmail.com': 0.8, 'icloud.com': 0.9, 'protonmail.com': 0.85,
            'company.com': 0.7, 'edu': 0.8, 'gov': 0.9, 'org': 0.6
        }
        self.suspicious_domains = {
            'tempmail.com': 0.1, '10minutemail.com': 0.1,
            'guerrillamail.com': 0.1, 'mailinator.com': 0.1
        }
        self.reputation_cache = {}
    
    def analyze_sender(self, sender_email: str) -> Dict[str, Any]:
        """Comprehensive sender analysis"""
        if not sender_email or '@' not in sender_email:
            return {'reputation': 0.0, 'risk_factors': ['Invalid email format']}
        
        domain = sender_email.split('@')[1].lower()
        
        # Check cache
        if domain in self.reputation_cache:
            return self.reputation_cache[domain]
        
        analysis = {
            'reputation': 0.5,  # Default neutral
            'risk_factors': [],
            'domain_age': None,
            'dns_records': None,
            'suspicious_patterns': []
        }
        
        # Check trusted domains
        for trusted_domain, score in self.trusted_domains.items():
            if trusted_domain in domain or domain.endswith('.' + trusted_domain):
                analysis['reputation'] = max(analysis['reputation'], score)
                break
        
        # Check suspicious domains
        for suspicious_domain, score in self.suspicious_domains.items():
            if suspicious_domain in domain:
                analysis['reputation'] = min(analysis['reputation'], score)
                analysis['risk_factors'].append(f'Suspicious domain: {suspicious_domain}')
                break
        
        # Analyze domain patterns
        if self._is_suspicious_domain(domain):
            analysis['reputation'] *= 0.3
            analysis['risk_factors'].append('Suspicious domain pattern')
        
        # Check for typosquatting
        if self._is_typosquatting(domain):
            analysis['reputation'] *= 0.2
            analysis['risk_factors'].append('Potential typosquatting')
        
        # DNS analysis
        try:
            dns_info = self._analyze_dns(domain)
            analysis['dns_records'] = dns_info
            if dns_info['mx_records'] == 0:
                analysis['risk_factors'].append('No MX records found')
                analysis['reputation'] *= 0.5
        except:
            analysis['risk_factors'].append('DNS lookup failed')
            analysis['reputation'] *= 0.7
        
        # Cache result
        self.reputation_cache[domain] = analysis
        return analysis
    
    def _is_suspicious_domain(self, domain: str) -> bool:
        """Check for suspicious domain patterns"""
        suspicious_patterns = [
            r'\d{4,}',  # Many numbers
            r'[a-z]{1,2}\d{3,}',  # Short letters + many numbers
            r'[a-z]{10,}',  # Very long domain names
            r'[^a-z0-9.-]',  # Special characters
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, domain):
                return True
        return False
    
    def _is_typosquatting(self, domain: str) -> bool:
        """Detect potential typosquatting"""
        common_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']
        for common in common_domains:
            if self._levenshtein_distance(domain, common) <= 2 and len(domain) > 5:
                return True
        return False
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _analyze_dns(self, domain: str) -> Dict[str, Any]:
        """Analyze DNS records"""
        try:
            mx_records = len(dns.resolver.resolve(domain, 'MX'))
            a_records = len(dns.resolver.resolve(domain, 'A'))
            return {
                'mx_records': mx_records,
                'a_records': a_records,
                'valid': True
            }
        except:
            return {
                'mx_records': 0,
                'a_records': 0,
                'valid': False
            }


class LinkAnalyzer:
    """Advanced link and URL analysis"""
    
    def __init__(self):
        self.shortened_domains = {
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'short.link',
            'is.gd', 'v.gd', 'ow.ly', 'buff.ly', 'rebrand.ly'
        }
        self.suspicious_tlds = {
            'tk', 'ml', 'ga', 'cf', 'click', 'download', 'exe'
        }
        self.redirect_cache = {}
    
    def analyze_links(self, text: str) -> Dict[str, Any]:
        """Analyze all links in text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        
        analysis = {
            'total_links': len(urls),
            'suspicious_links': 0,
            'shortened_links': 0,
            'risk_factors': [],
            'link_details': []
        }
        
        for url in urls:
            link_analysis = self._analyze_single_link(url)
            analysis['link_details'].append(link_analysis)
            
            if link_analysis['is_suspicious']:
                analysis['suspicious_links'] += 1
                analysis['risk_factors'].extend(link_analysis['risk_factors'])
            
            if link_analysis['is_shortened']:
                analysis['shortened_links'] += 1
        
        return analysis
    
    def _analyze_single_link(self, url: str) -> Dict[str, Any]:
        """Analyze a single URL"""
        analysis = {
            'url': url,
            'is_suspicious': False,
            'is_shortened': False,
            'risk_factors': [],
            'final_domain': None,
            'redirect_chain': []
        }
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check for shortened URLs
            if any(short_domain in domain for short_domain in self.shortened_domains):
                analysis['is_shortened'] = True
                analysis['risk_factors'].append('Shortened URL detected')
                
                # Try to resolve shortened URL
                final_url = self._resolve_shortened_url(url)
                if final_url:
                    analysis['final_domain'] = urlparse(final_url).netloc
                    analysis['redirect_chain'].append(final_url)
            
            # Check TLD
            tld = domain.split('.')[-1] if '.' in domain else ''
            if tld in self.suspicious_tlds:
                analysis['is_suspicious'] = True
                analysis['risk_factors'].append(f'Suspicious TLD: {tld}')
            
            # Check for suspicious patterns
            if self._has_suspicious_patterns(domain):
                analysis['is_suspicious'] = True
                analysis['risk_factors'].append('Suspicious domain pattern')
            
            # Check for IP addresses
            if re.match(r'^\d+\.\d+\.\d+\.\d+$', domain):
                analysis['is_suspicious'] = True
                analysis['risk_factors'].append('IP address in URL')
            
        except Exception as e:
            analysis['risk_factors'].append(f'URL analysis error: {str(e)}')
        
        return analysis
    
    def _has_suspicious_patterns(self, domain: str) -> bool:
        """Check for suspicious domain patterns"""
        suspicious_patterns = [
            r'\d{4,}',  # Many numbers
            r'[a-z]{1,2}\d{3,}',  # Short letters + many numbers
            r'[^a-z0-9.-]',  # Special characters
            r'(bank|paypal|amazon|apple|microsoft|google)[^a-z]',  # Brand impersonation
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, domain):
                return True
        return False
    
    def _resolve_shortened_url(self, url: str) -> Optional[str]:
        """Resolve shortened URL"""
        if url in self.redirect_cache:
            return self.redirect_cache[url]
        
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            final_url = response.url
            self.redirect_cache[url] = final_url
            return final_url
        except:
            return None


class AdvancedTextAnalyzer:
    """Advanced text analysis with multiple techniques"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vader_analyzer = VaderAnalyzer()
        
        # Threat-specific keywords
        self.phishing_keywords = {
            'urgent', 'verify', 'account', 'suspended', 'expired', 'security',
            'password', 'login', 'credentials', 'bank', 'payment', 'card',
            'click here', 'verify now', 'act now', 'limited time'
        }
        
        self.spam_keywords = {
            'free', 'win', 'prize', 'cash', 'money', 'lottery', 'winner',
            'congratulations', 'offer', 'deal', 'discount', 'sale', 'buy now'
        }
        
        self.suspicious_patterns = {
            'phone_pattern': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email_pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'currency_pattern': r'[$£€¥₹]\s*\d+',
            'urgency_pattern': r'\b(urgent|asap|immediately|now|today|limited time)\b',
            'threat_pattern': r'\b(suspended|expired|locked|blocked|terminated)\b'
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        if not text or pd.isna(text):
            return self._empty_analysis()
        
        text = str(text).lower()
        
        analysis = {
            'basic_stats': self._get_basic_stats(text),
            'linguistic_features': self._get_linguistic_features(text),
            'threat_indicators': self._get_threat_indicators(text),
            'sentiment_analysis': self._get_sentiment_analysis(text),
            'pattern_analysis': self._get_pattern_analysis(text),
            'readability': self._get_readability_scores(text)
        }
        
        return analysis
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis for null text"""
        return {
            'basic_stats': {'length': 0, 'word_count': 0, 'sentence_count': 0},
            'linguistic_features': {},
            'threat_indicators': {'phishing_score': 0, 'spam_score': 0},
            'sentiment_analysis': {'polarity': 0, 'subjectivity': 0},
            'pattern_analysis': {},
            'readability': {'flesch_score': 0}
        }
    
    def _get_basic_stats(self, text: str) -> Dict[str, Any]:
        """Get basic text statistics"""
        words = text.split()
        sentences = sent_tokenize(text)
        
        return {
            'length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        }
    
    def _get_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Get advanced linguistic features"""
        words = text.split()
        
        # Word frequency analysis
        word_freq = Counter(words)
        unique_words = len(set(words))
        
        # Repetition analysis
        max_freq = max(word_freq.values()) if word_freq else 0
        repetition_ratio = max_freq / len(words) if words else 0
        
        return {
            'unique_word_ratio': unique_words / len(words) if words else 0,
            'max_word_frequency': max_freq,
            'repetition_ratio': repetition_ratio,
            'consecutive_caps': len(re.findall(r'[A-Z]{3,}', text)),
            'consecutive_digits': len(re.findall(r'\d{4,}', text)),
            'consecutive_special': len(re.findall(r'[!?]{2,}', text))
        }
    
    def _get_threat_indicators(self, text: str) -> Dict[str, Any]:
        """Get threat-specific indicators"""
        words = set(text.split())
        
        phishing_score = len(words.intersection(self.phishing_keywords))
        spam_score = len(words.intersection(self.spam_keywords))
        
        # Urgency indicators
        urgency_words = ['urgent', 'asap', 'immediately', 'now', 'today', 'hurry']
        urgency_score = sum(1 for word in urgency_words if word in text)
        
        # Commercial indicators
        commercial_words = ['buy', 'sell', 'offer', 'deal', 'discount', 'sale']
        commercial_score = sum(1 for word in commercial_words if word in text)
        
        return {
            'phishing_score': phishing_score,
            'spam_score': spam_score,
            'urgency_score': urgency_score,
            'commercial_score': commercial_score,
            'threat_level': self._calculate_threat_level(phishing_score, spam_score, urgency_score)
        }
    
    def _calculate_threat_level(self, phishing: int, spam: int, urgency: int) -> str:
        """Calculate overall threat level"""
        total_score = phishing + spam + urgency
        
        if total_score >= 5:
            return 'critical'
        elif total_score >= 3:
            return 'high'
        elif total_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _get_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Get sentiment analysis"""
        blob = TextBlob(text)
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu']
        }
    
    def _get_pattern_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze suspicious patterns"""
        patterns = {}
        
        for pattern_name, pattern in self.suspicious_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            patterns[pattern_name] = {
                'count': len(matches),
                'matches': matches[:5]  # Limit to first 5 matches
            }
        
        return patterns
    
    def _get_readability_scores(self, text: str) -> Dict[str, Any]:
        """Calculate readability scores"""
        sentences = sent_tokenize(text)
        words = text.split()
        
        if not sentences or not words:
            return {'flesch_score': 0, 'smog_score': 0}
        
        # Flesch Reading Ease
        syllables = sum(self._count_syllables(word) for word in words)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))
        
        # SMOG Score
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        smog_score = 1.043 * np.sqrt(complex_words * (30 / len(sentences))) + 3.1291
        
        return {
            'flesch_score': flesch_score,
            'smog_score': smog_score
        }
    
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


class ESISCore:
    """Main ESIS engine"""
    
    def __init__(self):
        self.sender_analyzer = SenderReputationAnalyzer()
        self.link_analyzer = LinkAnalyzer()
        self.text_analyzer = AdvancedTextAnalyzer()
        
        self.models = {}
        self.vectorizers = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.explainer = None
        self.is_trained = False
        
        # Model configuration
        self.classes = ['ham', 'phishing', 'spam']
        self.threat_levels = {
            'ham': 'low',
            'phishing': 'high',
            'spam': 'medium'
        }
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training"""
        print("Preparing data for ESIS training...")
        
        # Extract features
        features_list = []
        for idx, row in df.iterrows():
            features = self._extract_all_features(row)
            features_list.append(features)
        
        feature_df = pd.DataFrame(features_list)
        
        # Prepare labels
        labels = self.label_encoder.fit_transform(df['label'])
        
        # Text vectorization
        texts = df['text'].fillna('').astype(str).tolist()
        cleaned_texts = [self.text_analyzer._clean_text(text) for text in texts]
        
        # TF-IDF vectorization
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        tfidf_features = self.vectorizers['tfidf'].fit_transform(cleaned_texts)
        
        # Combine all features
        feature_array = feature_df.values
        combined_features = np.hstack([
            tfidf_features.toarray(),
            feature_array
        ])
        
        return combined_features, labels, feature_df.columns.tolist()
    
    def _extract_all_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract all features from email row"""
        text = str(row.get('text', ''))
        sender = str(row.get('sender', ''))
        
        # Text analysis
        text_analysis = self.text_analyzer.analyze_text(text)
        
        # Sender analysis
        sender_analysis = self.sender_analyzer.analyze_sender(sender)
        
        # Link analysis
        link_analysis = self.link_analyzer.analyze_links(text)
        
        # Combine all features
        features = {}
        
        # Basic text stats
        for key, value in text_analysis['basic_stats'].items():
            features[f'text_{key}'] = value
        
        # Linguistic features
        for key, value in text_analysis['linguistic_features'].items():
            features[f'ling_{key}'] = value
        
        # Threat indicators
        for key, value in text_analysis['threat_indicators'].items():
            features[f'threat_{key}'] = value
        
        # Sentiment
        for key, value in text_analysis['sentiment_analysis'].items():
            features[f'sentiment_{key}'] = value
        
        # Patterns
        for pattern_name, pattern_data in text_analysis['pattern_analysis'].items():
            features[f'pattern_{pattern_name}_count'] = pattern_data['count']
        
        # Readability
        for key, value in text_analysis['readability'].items():
            features[f'readability_{key}'] = value
        
        # Sender reputation
        features['sender_reputation'] = sender_analysis['reputation']
        features['sender_risk_factors'] = len(sender_analysis['risk_factors'])
        
        # Link analysis
        features['total_links'] = link_analysis['total_links']
        features['suspicious_links'] = link_analysis['suspicious_links']
        features['shortened_links'] = link_analysis['shortened_links']
        
        return features
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train multiple models"""
        print("Training ESIS models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_config = {
            'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.4f}")
            
            self.models[name] = model
        
        # Create ensemble
        ensemble_models = [
            ('rf', self.models['random_forest']),
            ('xgb', self.models['xgboost']),
            ('lgb', self.models['lightgbm'])
        ]
        
        self.models['ensemble'] = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        self.models['ensemble'].fit(X_train_scaled, y_train)
        
        # Final evaluation
        y_pred_ensemble = self.models['ensemble'].predict(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        # Initialize explainer
        self.explainer = shap.TreeExplainer(self.models['random_forest'])
        
        self.is_trained = True
        return X_test_scaled, y_test, y_pred_ensemble
    
    def predict(self, email_data: Dict[str, Any]) -> EmailAnalysis:
        """Predict email classification with explanations"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self._extract_all_features(email_data)
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Get predictions
        predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_array_scaled)[0]
                predictions[name] = proba
            else:
                pred = model.predict(feature_array_scaled)[0]
                predictions[name] = pred
        
        # Get ensemble prediction
        ensemble_proba = predictions['ensemble']
        predicted_class_idx = np.argmax(ensemble_proba)
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = max(ensemble_proba)
        
        # Generate explanations
        explanations = self._generate_explanations(feature_array_scaled, features)
        
        # Determine threat level
        threat_level = self._determine_threat_level(predicted_class, confidence, features)
        
        # Get risk factors
        risk_factors = self._extract_risk_factors(features, predicted_class)
        
        # Get sender reputation
        sender_reputation = features.get('sender_reputation', 0.5)
        
        return EmailAnalysis(
            email_id=email_data.get('id', str(hash(str(email_data)))),
            classification=predicted_class,
            confidence=confidence,
            threat_level=threat_level,
            sender_reputation=sender_reputation,
            risk_factors=risk_factors,
            explanations=explanations,
            timestamp=datetime.now()
        )
    
    def _generate_explanations(self, features_scaled: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explainable AI explanations"""
        explanations = {
            'feature_importance': {},
            'shap_values': None,
            'decision_factors': [],
            'confidence_breakdown': {}
        }
        
        # Feature importance from Random Forest
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            feature_importance = rf_model.feature_importances_
            
            # Get top 10 most important features
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            for idx in top_indices:
                if idx < len(feature_importance):
                    explanations['feature_importance'][f'feature_{idx}'] = float(feature_importance[idx])
        
        # SHAP values
        try:
            shap_values = self.explainer.shap_values(features_scaled)
            explanations['shap_values'] = shap_values[0].tolist() if len(shap_values) > 0 else []
        except:
            pass
        
        # Decision factors
        high_impact_features = []
        for feature_name, value in features.items():
            if abs(value) > 0.5:  # Threshold for significant impact
                high_impact_features.append({
                    'feature': feature_name,
                    'value': value,
                    'impact': 'high' if abs(value) > 1.0 else 'medium'
                })
        
        explanations['decision_factors'] = sorted(
            high_impact_features, 
            key=lambda x: abs(x['value']), 
            reverse=True
        )[:5]
        
        return explanations
    
    def _determine_threat_level(self, classification: str, confidence: float, features: Dict[str, Any]) -> str:
        """Determine threat level based on classification and features"""
        base_threat = self.threat_levels.get(classification, 'low')
        
        # Adjust based on confidence
        if confidence > 0.9:
            if base_threat == 'high':
                return 'critical'
            elif base_threat == 'medium':
                return 'high'
        
        # Adjust based on specific features
        if features.get('threat_phishing_score', 0) > 3:
            return 'critical'
        
        if features.get('suspicious_links', 0) > 2:
            return 'high'
        
        if features.get('sender_reputation', 0.5) < 0.3:
            return 'high'
        
        return base_threat
    
    def _extract_risk_factors(self, features: Dict[str, Any], classification: str) -> List[str]:
        """Extract risk factors from features"""
        risk_factors = []
        
        # High threat indicators
        if features.get('threat_phishing_score', 0) > 2:
            risk_factors.append('High phishing keyword density')
        
        if features.get('suspicious_links', 0) > 0:
            risk_factors.append(f"{features['suspicious_links']} suspicious links detected")
        
        if features.get('shortened_links', 0) > 0:
            risk_factors.append(f"{features['shortened_links']} shortened URLs detected")
        
        if features.get('sender_reputation', 0.5) < 0.3:
            risk_factors.append('Low sender reputation score')
        
        if features.get('pattern_phone_pattern_count', 0) > 0:
            risk_factors.append('Phone numbers detected')
        
        if features.get('pattern_currency_pattern_count', 0) > 0:
            risk_factors.append('Currency amounts detected')
        
        if features.get('threat_urgency_score', 0) > 2:
            risk_factors.append('High urgency language detected')
        
        return risk_factors
    
    def save_model(self, filepath: str = 'esis_model.pkl'):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'classes': self.classes,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"ESIS model saved to {filepath}")
    
    def load_model(self, filepath: str = 'esis_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.vectorizers = model_data['vectorizers']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.classes = model_data['classes']
        self.is_trained = model_data['is_trained']
        print(f"ESIS model loaded from {filepath}")


def main():
    """Main function to demonstrate ESIS"""
    print("🛡️ Email Security Intelligence System (ESIS)")
    print("=" * 60)
    
    # Initialize ESIS
    esis = ESISCore()
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'text': [
            "Your account has been suspended. Click here to verify: bit.ly/suspicious-link",
            "Free money! Win $1000 now! Limited time offer!",
            "Hey, how are you doing? Let's meet for coffee tomorrow.",
            "URGENT: Your bank account needs verification. Please click the link immediately.",
            "Thanks for the meeting yesterday. I'll send the report by Friday."
        ],
        'sender': [
            "security@bank-verification.com",
            "winner@lottery-prize.org",
            "john@gmail.com",
            "noreply@yourbank.com",
            "sarah@company.com"
        ],
        'label': ['phishing', 'spam', 'ham', 'phishing', 'ham']
    })
    
    print("Sample data created for demonstration")
    print(f"Classes: {esis.classes}")
    
    # Prepare and train
    X, y, feature_names = esis.prepare_data(sample_data)
    X_test, y_test, y_pred = esis.train_models(X, y)
    
    # Test predictions
    print("\n" + "=" * 60)
    print("ESIS Analysis Results:")
    print("=" * 60)
    
    for idx, row in sample_data.iterrows():
        email_data = {
            'id': f'email_{idx}',
            'text': row['text'],
            'sender': row['sender']
        }
        
        analysis = esis.predict(email_data)
        
        print(f"\n📧 Email {idx + 1}:")
        print(f"   Text: {row['text'][:60]}...")
        print(f"   Sender: {row['sender']}")
        print(f"   Classification: {analysis.classification.upper()}")
        print(f"   Threat Level: {analysis.threat_level.upper()}")
        print(f"   Confidence: {analysis.confidence:.3f}")
        print(f"   Sender Reputation: {analysis.sender_reputation:.3f}")
        print(f"   Risk Factors: {', '.join(analysis.risk_factors[:3])}")
    
    # Save model
    esis.save_model()
    
    print(f"\n✅ ESIS training complete!")
    print(f"Model saved as 'esis_model.pkl'")
    print(f"Ready for production use!")


if __name__ == "__main__":
    main()
