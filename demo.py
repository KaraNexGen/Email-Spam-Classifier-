"""
Demo Script for Email Spam Classifier
====================================

This script demonstrates the capabilities of the Advanced Email Spam Classifier
with sample emails and real-time predictions.
"""

import pandas as pd
import numpy as np
from email_spam_classifier import AdvancedSpamClassifier
import time

def print_separator(title=""):
    """Print a formatted separator"""
    print("\n" + "=" * 60)
    if title:
        print(f" {title}")
        print("=" * 60)

def print_result(text, prediction, confidence, features=None):
    """Print formatted prediction result"""
    print(f"\n📧 Email Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"🎯 Classification: {'🚨 SPAM' if prediction > 0.5 else '✅ HAM'}")
    print(f"📊 Spam Probability: {prediction:.4f}")
    print(f"🎲 Confidence: {confidence:.4f}")
    
    if features:
        print(f"🔍 Key Features:")
        for feature, value in list(features.items())[:5]:
            print(f"   • {feature}: {value:.3f}")

def main():
    """Main demo function"""
    print_separator("🚀 Advanced Email Spam Classifier Demo")
    
    # Sample emails for demonstration
    sample_emails = [
        {
            "text": "Hey, how are you doing? Let's meet for coffee tomorrow at 3 PM. I have some exciting news to share with you!",
            "expected": "HAM"
        },
        {
            "text": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
            "expected": "SPAM"
        },
        {
            "text": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
            "expected": "SPAM"
        },
        {
            "text": "I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.",
            "expected": "HAM"
        },
        {
            "text": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
            "expected": "SPAM"
        },
        {
            "text": "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
            "expected": "HAM"
        },
        {
            "text": "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info",
            "expected": "SPAM"
        },
        {
            "text": "I HAVE A DATE ON SUNDAY WITH WILL!!",
            "expected": "HAM"
        }
    ]
    
    print("📋 Loading and training the model...")
    print("⏳ This may take a few minutes for the first run...")
    
    try:
        # Initialize classifier
        classifier = AdvancedSpamClassifier()
        
        # Load and prepare data
        print("\n📊 Loading dataset...")
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']].dropna()
        df.columns = ['label', 'text']
        
        print(f"✅ Dataset loaded: {df.shape[0]} emails")
        print(f"📈 Spam: {len(df[df['label'] == 'spam'])} ({len(df[df['label'] == 'spam'])/len(df)*100:.1f}%)")
        print(f"📈 Ham: {len(df[df['label'] == 'ham'])} ({len(df[df['label'] == 'ham'])/len(df)*100:.1f}%)")
        
        # Prepare data and train
        print("\n🔧 Preparing features...")
        X, y, feature_names = classifier.prepare_data(df)
        print(f"✅ Features extracted: {X.shape[1]} features")
        
        print("\n🤖 Training models...")
        start_time = time.time()
        X_test, y_test, y_pred = classifier.train_models(X, y)
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Test with sample emails
        print_separator("🧪 Testing with Sample Emails")
        
        correct_predictions = 0
        total_predictions = len(sample_emails)
        
        for i, email in enumerate(sample_emails, 1):
            print(f"\n--- Test {i}/{total_predictions} ---")
            
            # Make prediction
            predictions = classifier.predict([email['text']])
            ensemble_prob = predictions['ensemble'][0]
            confidence = max(ensemble_prob, 1 - ensemble_prob)
            
            # Extract features for analysis
            features_df = classifier.feature_extractor.extract_all_features([email['text']])
            top_features = features_df.iloc[0].sort_values(ascending=False).head(5).to_dict()
            
            # Print result
            print_result(email['text'], ensemble_prob, confidence, top_features)
            
            # Check if prediction is correct
            predicted_class = "SPAM" if ensemble_prob > 0.5 else "HAM"
            if predicted_class == email['expected']:
                correct_predictions += 1
                print("✅ Correct prediction!")
            else:
                print(f"❌ Incorrect prediction! Expected: {email['expected']}")
        
        # Summary
        print_separator("📊 Demo Summary")
        accuracy = correct_predictions / total_predictions
        print(f"🎯 Demo Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
        print(f"⏱️  Training Time: {training_time:.2f} seconds")
        print(f"🔧 Features Used: {X.shape[1]}")
        print(f"🤖 Models Trained: {len(classifier.models)}")
        
        # Show model performance
        print(f"\n📈 Model Performance:")
        for name, model in classifier.models.items():
            if hasattr(model, 'predict_proba'):
                # Quick test on a sample
                test_pred = classifier.predict(["This is a test email"])
                prob = test_pred['ensemble'][0]
                print(f"   • {name}: Ready ✅")
        
        # Save model
        print(f"\n💾 Saving trained model...")
        classifier.save_model('spam_classifier_model.pkl')
        print("✅ Model saved successfully!")
        
        # Interactive testing
        print_separator("🎮 Interactive Testing")
        print("Enter your own email text to test the classifier!")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input("\n📧 Enter email text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                print("❌ Please enter some text!")
                continue
            
            try:
                # Make prediction
                predictions = classifier.predict([user_input])
                ensemble_prob = predictions['ensemble'][0]
                confidence = max(ensemble_prob, 1 - ensemble_prob)
                
                # Extract features
                features_df = classifier.feature_extractor.extract_all_features([user_input])
                top_features = features_df.iloc[0].sort_values(ascending=False).head(5).to_dict()
                
                print_result(user_input, ensemble_prob, confidence, top_features)
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print_separator("👋 Demo Complete!")
        print("Thank you for trying the Advanced Email Spam Classifier!")
        print("\nNext steps:")
        print("• Run 'streamlit run visualization_dashboard.py' for interactive dashboard")
        print("• Run 'python api_server.py' to start the API server")
        print("• Run 'python train_and_evaluate.py' for comprehensive evaluation")
        
    except FileNotFoundError:
        print("❌ Error: spam.csv file not found!")
        print("Please ensure the dataset file is in the current directory.")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()
