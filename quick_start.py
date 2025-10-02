"""
Quick Start Email Spam Classifier
================================
Minimal setup to get the spam classifier running immediately.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
import re
import string
from collections import Counter
import joblib

class QuickSpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {}
        self.is_trained = False
    
    def clean_text(self, text):
        """Quick text cleaning"""
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_features(self, texts):
        """Extract basic features"""
        features = []
        for text in texts:
            text = self.clean_text(text)
            features.append({
                'length': len(text),
                'word_count': len(text.split()),
                'exclamation': text.count('!'),
                'caps_words': sum(1 for word in text.split() if word.isupper() and len(word) > 1),
                'spam_words': sum(1 for word in ['free', 'win', 'prize', 'urgent', 'cash', 'money'] if word in text),
                'has_phone': 1 if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text) else 0,
                'has_url': 1 if re.search(r'http', text) else 0
            })
        return pd.DataFrame(features)
    
    def train(self, df):
        """Train the classifier"""
        print("Training spam classifier...")
        
        # Prepare data
        texts = df['text'].fillna('').astype(str).tolist()
        labels = (df['label'] == 'spam').astype(int).values
        
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(cleaned_texts)
        
        # Additional features
        additional_features = self.extract_features(texts)
        
        # Combine features
        from scipy.sparse import hstack
        X = hstack([tfidf_features, additional_features.values])
        y = labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        self.models['naive_bayes'] = MultinomialNB()
        self.models['logistic_regression'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train each model
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name}: {accuracy:.4f}")
        
        # Create ensemble
        self.models['ensemble'] = VotingClassifier([
            ('nb', self.models['naive_bayes']),
            ('lr', self.models['logistic_regression']),
            ('rf', self.models['random_forest'])
        ], voting='soft')
        
        self.models['ensemble'].fit(X_train, y_train)
        y_pred_ensemble = self.models['ensemble'].predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        print(f"Ensemble: {ensemble_accuracy:.4f}")
        
        self.is_trained = True
        return X_test, y_test, y_pred_ensemble
    
    def predict(self, text):
        """Predict if text is spam"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Clean and prepare text
        cleaned_text = self.clean_text(text)
        tfidf_features = self.vectorizer.transform([cleaned_text])
        additional_features = self.extract_features([text])
        
        # Combine features
        from scipy.sparse import hstack
        X = hstack([tfidf_features, additional_features.values])
        
        # Get ensemble prediction
        prob = self.models['ensemble'].predict_proba(X)[0][1]
        classification = "SPAM" if prob > 0.5 else "HAM"
        
        return {
            'text': text,
            'classification': classification,
            'spam_probability': prob,
            'confidence': max(prob, 1 - prob)
        }
    
    def save_model(self, filepath='quick_spam_model.pkl'):
        """Save the trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='quick_spam_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.models = model_data['models']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")

def main():
    """Quick start main function"""
    print("Quick Email Spam Classifier")
    print("=" * 40)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].dropna()
    df.columns = ['label', 'text']
    
    print(f"Dataset: {df.shape[0]} emails")
    print(f"Spam: {len(df[df['label'] == 'spam'])} ({len(df[df['label'] == 'spam'])/len(df)*100:.1f}%)")
    
    # Train classifier
    classifier = QuickSpamClassifier()
    X_test, y_test, y_pred = classifier.train(df)
    
    # Test with sample emails
    print("\n" + "=" * 40)
    print("Testing with sample emails:")
    print("=" * 40)
    
    test_emails = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts!",
        "Hey, how are you doing? Let's meet for coffee tomorrow.",
        "URGENT! You have won a prize! Call now!",
        "Thanks for the meeting yesterday. Let's follow up next week."
    ]
    
    for email in test_emails:
        result = classifier.predict(email)
        print(f"\nEmail: {email[:50]}...")
        print(f"Result: {result['classification']} (Prob: {result['spam_probability']:.3f})")
    
    # Save model
    classifier.save_model()
    
    print("\nQuick setup complete!")
    print("Model saved as 'quick_spam_model.pkl'")
    print("\nTo use the model:")
    print("classifier = QuickSpamClassifier()")
    print("classifier.load_model()")
    print("result = classifier.predict('your email text')")

if __name__ == "__main__":
    main()
