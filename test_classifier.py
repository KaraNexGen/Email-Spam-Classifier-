"""
Quick Test Script for Spam Classifier
====================================
Test the classifier with sample emails.
"""

from quick_start import QuickSpamClassifier

def test_classifier():
    # Load the trained model
    classifier = QuickSpamClassifier()
    classifier.load_model('quick_spam_model.pkl')
    
    print("Email Spam Classifier - Test Results")
    print("=" * 50)
    
    # Test emails
    test_emails = [
        "Free money! Click here to win $1000!",
        "Hey, how are you doing today?",
        "URGENT! You have won a prize! Call now!",
        "Thanks for the meeting yesterday.",
        "Congratulations! You've been selected for a free iPhone!",
        "Can we schedule a call for tomorrow?",
        "WINNER! Claim your prize now! Limited time offer!",
        "I'll see you at the office tomorrow morning."
    ]
    
    for i, email in enumerate(test_emails, 1):
        result = classifier.predict(email)
        print(f"\n{i}. Email: {email}")
        print(f"   Result: {result['classification']}")
        print(f"   Spam Probability: {result['spam_probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    test_classifier()
