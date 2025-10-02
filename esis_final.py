"""
ESIS Final Version - Working Email Security System
================================================
A working version of ESIS that avoids all the scaling issues
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
import joblib
from collections import Counter

# Basic ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

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


class SimpleTextAnalyzer:
    """Simplified text analyzer"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Threat-specific keywords
        self.phishing_keywords = {
            'urgent', 'verify', 'account', 'suspended', 'expired', 'security',
            'password', 'login', 'credentials', 'bank', 'payment', 'card',
            'click here', 'verify now', 'act now', 'limited time', 'immediately'
        }
        
        self.spam_keywords = {
            'free', 'win', 'prize', 'cash', 'money', 'lottery', 'winner',
            'congratulations', 'offer', 'deal', 'discount', 'sale', 'buy now',
            'limited time', 'act now', 'call now', 'text now'
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        if not text or pd.isna(text):
            return self._empty_analysis()
        
        text = str(text).lower()
        
        analysis = {
            'basic_stats': self._get_basic_stats(text),
            'threat_indicators': self._get_threat_indicators(text),
            'sentiment_analysis': self._get_sentiment_analysis(text),
            'pattern_analysis': self._get_pattern_analysis(text)
        }
        
        return analysis
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis for null text"""
        return {
            'basic_stats': {'length': 0, 'word_count': 0, 'sentence_count': 0},
            'threat_indicators': {'phishing_score': 0, 'spam_score': 0},
            'sentiment_analysis': {'polarity': 0, 'subjectivity': 0},
            'pattern_analysis': {}
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
            'commercial_score': commercial_score
        }
    
    def _get_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Get sentiment analysis"""
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def _get_pattern_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze suspicious patterns"""
        patterns = {
            'phone_pattern': len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)),
            'email_pattern': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            'currency_pattern': len(re.findall(r'[$£€¥₹]\s*\d+', text)),
            'urgency_pattern': len(re.findall(r'\b(urgent|asap|immediately|now|today|limited time)\b', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_word_count': len(re.findall(r'\b[A-Z]{3,}\b', text))
        }
        
        return patterns


class SimpleSenderAnalyzer:
    """Simplified sender reputation analyzer"""
    
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
    
    def analyze_sender(self, sender_email: str) -> Dict[str, Any]:
        """Analyze sender reputation"""
        if not sender_email or '@' not in sender_email:
            return {'reputation': 0.0, 'risk_factors': ['Invalid email format']}
        
        domain = sender_email.split('@')[1].lower()
        
        analysis = {
            'reputation': 0.5,  # Default neutral
            'risk_factors': []
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
        
        # Check for suspicious patterns
        if self._is_suspicious_domain(domain):
            analysis['reputation'] *= 0.3
            analysis['risk_factors'].append('Suspicious domain pattern')
        
        return analysis
    
    def _is_suspicious_domain(self, domain: str) -> bool:
        """Check for suspicious domain patterns"""
        suspicious_patterns = [
            r'\d{4,}',  # Many numbers
            r'[a-z]{1,2}\d{3,}',  # Short letters + many numbers
            r'[a-z]{10,}',  # Very long domain names
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, domain):
                return True
        return False


class SimpleESIS:
    """Simplified ESIS system"""
    
    def __init__(self):
        self.text_analyzer = SimpleTextAnalyzer()
        self.sender_analyzer = SimpleSenderAnalyzer()
        
        self.models = {}
        self.vectorizers = {}
        self.label_encoder = LabelEncoder()
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
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # TF-IDF vectorization
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        tfidf_features = self.vectorizers['tfidf'].fit_transform(cleaned_texts)
        
        # Combine all features
        feature_array = feature_df.values.astype(float)
        
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
        
        # Combine all features
        features = {}
        
        # Basic text stats
        for key, value in text_analysis['basic_stats'].items():
            features[f'text_{key}'] = float(value) if isinstance(value, (int, float)) else 0.0
        
        # Threat indicators
        for key, value in text_analysis['threat_indicators'].items():
            features[f'threat_{key}'] = float(value) if isinstance(value, (int, float)) else 0.0
        
        # Sentiment
        for key, value in text_analysis['sentiment_analysis'].items():
            features[f'sentiment_{key}'] = float(value) if isinstance(value, (int, float)) else 0.0
        
        # Patterns
        for pattern_name, pattern_value in text_analysis['pattern_analysis'].items():
            features[f'pattern_{pattern_name}'] = float(pattern_value) if isinstance(pattern_value, (int, float)) else 0.0
        
        # Sender reputation
        features['sender_reputation'] = float(sender_analysis['reputation'])
        features['sender_risk_factors'] = float(len(sender_analysis['risk_factors']))
        
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
        
        # Define models - use only models that work well together
        models_config = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.4f}")
            
            self.models[name] = model
        
        # Create ensemble
        self.models['ensemble'] = VotingClassifier(
            estimators=[
                ('lr', self.models['logistic_regression']),
                ('rf', self.models['random_forest'])
            ],
            voting='soft'
        )
        
        self.models['ensemble'].fit(X_train, y_train)
        
        # Final evaluation
        y_pred_ensemble = self.models['ensemble'].predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        self.is_trained = True
        return X_test, y_test, y_pred_ensemble
    
    def predict(self, email_data: Dict[str, Any]) -> EmailAnalysis:
        """Predict email classification with explanations"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features the same way as training
        features = self._extract_all_features(email_data)
        
        # Get text for TF-IDF
        text = str(email_data.get('text', ''))
        cleaned_text = self._clean_text(text)
        
        # Transform text with TF-IDF
        tfidf_features = self.vectorizers['tfidf'].transform([cleaned_text])
        
        # Get feature array
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Combine TF-IDF and other features
        combined_features = np.hstack([
            tfidf_features.toarray(),
            feature_array
        ])
        
        # Get predictions
        predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(combined_features)[0]
                predictions[name] = proba
            else:
                pred = model.predict(combined_features)[0]
                predictions[name] = pred
        
        # Get ensemble prediction
        ensemble_proba = predictions['ensemble']
        predicted_class_idx = np.argmax(ensemble_proba)
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = max(ensemble_proba)
        
        # Generate explanations
        explanations = self._generate_explanations(combined_features, features)
        
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
    
    def _generate_explanations(self, features_array: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explainable AI explanations"""
        explanations = {
            'feature_importance': {},
            'decision_factors': []
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
        
        if features.get('threat_spam_score', 0) > 3:
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
        
        if features.get('threat_spam_score', 0) > 2:
            risk_factors.append('High spam keyword density')
        
        if features.get('sender_reputation', 0.5) < 0.3:
            risk_factors.append('Low sender reputation score')
        
        if features.get('pattern_phone_pattern', 0) > 0:
            risk_factors.append('Phone numbers detected')
        
        if features.get('pattern_currency_pattern', 0) > 0:
            risk_factors.append('Currency amounts detected')
        
        if features.get('threat_urgency_score', 0) > 2:
            risk_factors.append('High urgency language detected')
        
        if features.get('pattern_exclamation_count', 0) > 3:
            risk_factors.append('Excessive exclamation marks')
        
        return risk_factors
    
    def save_model(self, filepath: str = 'esis_final_model.pkl'):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'label_encoder': self.label_encoder,
            'classes': self.classes,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"ESIS model saved to {filepath}")
    
    def load_model(self, filepath: str = 'esis_final_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.vectorizers = model_data['vectorizers']
        self.label_encoder = model_data['label_encoder']
        self.classes = model_data['classes']
        self.is_trained = model_data['is_trained']
        print(f"ESIS model loaded from {filepath}")


def create_demo_dataset():
    """Create demo dataset"""
    print("Creating demo dataset...")
    
    # Phishing emails
    phishing_emails = [
        "URGENT: Your account has been suspended due to suspicious activity. Click here to verify your identity immediately: bit.ly/suspicious-link. Failure to verify within 24 hours will result in permanent account closure.",
        "Security Alert: Unusual login activity detected from a new device. Verify your account to prevent unauthorized access: https://secure-bank-verification.com/login",
        "Your PayPal account has been limited due to suspicious transactions. Click here to restore access: paypal-security-verification.net/restore",
        "Amazon: Your order cannot be processed. Update your payment information immediately to avoid cancellation: amazon-payment-update.com/verify",
        "Microsoft: Your account will be closed due to policy violations. Verify your identity now: microsoft-account-verification.org/secure"
    ]
    
    # Spam emails
    spam_emails = [
        "Free money! Win $1000 now! Limited time offer! Click here to claim: bit.ly/free-money",
        "Congratulations! You've won a free iPhone! Claim your prize now: prize-claim.com/iphone",
        "URGENT: You have won a prize! Call now to claim your reward: 1-800-WIN-PRIZE",
        "Free entry in 2 a wkly comp to win FA Cup final tkts! Text WIN to 12345",
        "SIX chances to win CASH! From 100 to 20,000 pounds! Reply CASH to claim"
    ]
    
    # Ham emails
    ham_emails = [
        "Hey, how are you doing? Let's meet for coffee tomorrow at 3 PM.",
        "Thanks for the meeting yesterday. I'll send the report by Friday.",
        "Can we schedule a call for tomorrow morning? I have some questions about the project.",
        "I'll see you at the office tomorrow morning. Don't forget to bring the documents.",
        "Thanks for your help with the project. Great work on the presentation!"
    ]
    
    # Senders
    phishing_senders = [
        "security@bank-verification.com", "alerts@security-center.org", "noreply@yourbank.com",
        "support@paypal-verification.net", "orders@amazon-security.com"
    ]
    
    spam_senders = [
        "winner@lottery-prize.org", "prizes@free-iphone.com", "rewards@prize-claim.net",
        "competitions@free-tickets.org", "cash@win-money.com"
    ]
    
    ham_senders = [
        "john@gmail.com", "sarah@company.com", "mike@business.org", "lisa@office.com", "team@project.net"
    ]
    
    # Create dataset
    data = []
    
    for i, email in enumerate(phishing_emails):
        data.append({
            'text': email,
            'sender': phishing_senders[i % len(phishing_senders)],
            'label': 'phishing'
        })
    
    for i, email in enumerate(spam_emails):
        data.append({
            'text': email,
            'sender': spam_senders[i % len(spam_senders)],
            'label': 'spam'
        })
    
    for i, email in enumerate(ham_emails):
        data.append({
            'text': email,
            'sender': ham_senders[i % len(ham_senders)],
            'label': 'ham'
        })
    
    return pd.DataFrame(data)


def main():
    """Main function to demonstrate ESIS"""
    print("Email Security Intelligence System (ESIS) - Final Version")
    print("=" * 70)
    
    # Create demo dataset
    df = create_demo_dataset()
    print(f"Dataset created: {len(df)} emails")
    print(f"Phishing: {len(df[df['label'] == 'phishing'])}")
    print(f"Spam: {len(df[df['label'] == 'spam'])}")
    print(f"Ham: {len(df[df['label'] == 'ham'])}")
    
    # Initialize ESIS
    esis = SimpleESIS()
    
    # Prepare and train
    X, y, feature_names = esis.prepare_data(df)
    X_test, y_test, y_pred = esis.train_models(X, y)
    
    print(f"\nFeatures extracted: {X.shape[1]}")
    
    # Test with sample emails
    print("\n" + "=" * 50)
    print("ESIS Analysis Results:")
    print("=" * 50)
    
    test_emails = [
        {
            'text': "URGENT: Your account has been suspended. Click here to verify: bit.ly/suspicious-link",
            'sender': "security@bank-verification.com",
            'expected': 'phishing'
        },
        {
            'text': "Free money! Win $1000 now! Limited time offer!",
            'sender': "winner@lottery-prize.org",
            'expected': 'spam'
        },
        {
            'text': "Hey, how are you doing? Let's meet for coffee tomorrow.",
            'sender': "john@gmail.com",
            'expected': 'ham'
        }
    ]
    
    for i, email_data in enumerate(test_emails, 1):
        analysis = esis.predict(email_data)
        
        print(f"\nTest {i}:")
        print(f"Text: {email_data['text'][:60]}...")
        print(f"Sender: {email_data['sender']}")
        print(f"Expected: {email_data['expected']}")
        print(f"Predicted: {analysis.classification}")
        print(f"Threat Level: {analysis.threat_level}")
        print(f"Confidence: {analysis.confidence:.3f}")
        print(f"Sender Reputation: {analysis.sender_reputation:.3f}")
        print(f"Risk Factors: {', '.join(analysis.risk_factors[:3])}")
        
        # Check if prediction is correct
        if analysis.classification == email_data['expected']:
            print("Correct prediction!")
        else:
            print("Incorrect prediction!")
    
    # Save model
    esis.save_model()
    
    print(f"\nESIS training complete!")
    print(f"Model saved as 'esis_final_model.pkl'")
    print(f"Ready for production use!")


if __name__ == "__main__":
    main()
