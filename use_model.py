"""
Use the trained spam classifier
==============================
Simple script to use the trained model for predictions.
"""

from quick_start import QuickSpamClassifier

def main():
    # Load the trained model
    classifier = QuickSpamClassifier()
    classifier.load_model('quick_spam_model.pkl')
    
    print("Email Spam Classifier - Ready!")
    print("Enter email text to classify (type 'quit' to exit):")
    print("-" * 50)
    
    while True:
        email_text = input("\nEmail: ").strip()
        
        if email_text.lower() in ['quit', 'exit', 'q']:
            break
            
        if not email_text:
            print("Please enter some text!")
            continue
            
        try:
            result = classifier.predict(email_text)
            print(f"Classification: {result['classification']}")
            print(f"Spam Probability: {result['spam_probability']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
