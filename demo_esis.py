"""
ESIS Demo Script
===============
Demonstration of Email Security Intelligence System capabilities
"""

import pandas as pd
import numpy as np
from esis_core import ESISCore
import json
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_analysis(analysis, email_text, sender):
    """Print formatted analysis results"""
    print(f"\n📧 Email: {email_text[:60]}{'...' if len(email_text) > 60 else ''}")
    print(f"👤 Sender: {sender}")
    print(f"🎯 Classification: {analysis.classification.upper()}")
    print(f"⚠️  Threat Level: {analysis.threat_level.upper()}")
    print(f"📊 Confidence: {analysis.confidence:.3f}")
    print(f"🏆 Sender Reputation: {analysis.sender_reputation:.3f}")
    print(f"🔍 Risk Factors: {', '.join(analysis.risk_factors[:3])}")
    
    # Show explanations
    if analysis.explanations and 'decision_factors' in analysis.explanations:
        print(f"💡 Key Decision Factors:")
        for factor in analysis.explanations['decision_factors'][:3]:
            print(f"   • {factor['feature']}: {factor['value']:.3f} ({factor['impact']} impact)")

def create_demo_dataset():
    """Create demo dataset for training"""
    print("Creating comprehensive demo dataset...")
    
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

def demonstrate_esis():
    """Demonstrate ESIS capabilities"""
    print_header("🛡️ Email Security Intelligence System (ESIS) Demo")
    print("Advanced Multi-Threat Email Analysis & Adaptive Defense Platform")
    
    # Create demo dataset
    df = create_demo_dataset()
    print(f"\n📊 Dataset created: {len(df)} emails")
    print(f"   • Phishing: {len(df[df['label'] == 'phishing'])}")
    print(f"   • Spam: {len(df[df['label'] == 'spam'])}")
    print(f"   • Ham: {len(df[df['label'] == 'ham'])}")
    
    # Initialize ESIS
    print("\n🤖 Initializing ESIS...")
    esis = ESISCore()
    
    # Prepare and train
    print("🔧 Preparing data and training models...")
    X, y, feature_names = esis.prepare_data(df)
    X_test, y_test, y_pred = esis.train_models(X, y)
    
    print(f"✅ Training complete! Features: {X.shape[1]}")
    
    # Test with various email types
    print_header("🧪 ESIS Analysis Results")
    
    test_cases = [
        {
            'text': "URGENT: Your account has been suspended. Click here to verify: bit.ly/suspicious-link",
            'sender': "security@bank-verification.com",
            'type': 'Phishing'
        },
        {
            'text': "Free money! Win $1000 now! Limited time offer!",
            'sender': "winner@lottery-prize.org",
            'type': 'Spam'
        },
        {
            'text': "Hey, how are you doing? Let's meet for coffee tomorrow.",
            'sender': "john@gmail.com",
            'type': 'Legitimate'
        },
        {
            'text': "Your bank account needs immediate verification. Please click the link to avoid closure.",
            'sender': "noreply@yourbank.com",
            'type': 'Phishing'
        },
        {
            'text': "Congratulations! You've won a free iPhone! Claim your prize now!",
            'sender': "prizes@free-iphone.com",
            'type': 'Spam'
        },
        {
            'text': "Thanks for the meeting yesterday. I'll send the report by Friday.",
            'sender': "sarah@company.com",
            'type': 'Legitimate'
        }
    ]
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['type']} ---")
        
        email_data = {
            'id': f'test_{i}',
            'text': test_case['text'],
            'sender': test_case['sender']
        }
        
        try:
            analysis = esis.predict(email_data)
            print_analysis(analysis, test_case['text'], test_case['sender'])
            
            # Check if prediction matches expected type
            expected_class = test_case['type'].lower()
            if expected_class == 'legitimate':
                expected_class = 'ham'
            
            if analysis.classification == expected_class:
                print("✅ Correct prediction!")
                correct_predictions += 1
            else:
                print(f"❌ Incorrect prediction! Expected: {expected_class}")
                
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
    
    # Summary
    print_header("📊 Demo Summary")
    accuracy = correct_predictions / total_predictions
    print(f"🎯 Demo Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    print(f"🔧 Features Used: {X.shape[1]}")
    print(f"🤖 Models Trained: {len(esis.models)}")
    print(f"📈 System Status: Ready for production use!")
    
    # Save model
    esis.save_model('esis_demo_model.pkl')
    print(f"💾 Model saved as 'esis_demo_model.pkl'")
    
    # Interactive testing
    print_header("🎮 Interactive Testing")
    print("Enter your own email text to test ESIS!")
    print("Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\n📧 Enter email text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                print("❌ Please enter some text!")
                continue
            
            sender = input("👤 Enter sender email (or press Enter for default): ").strip()
            if not sender:
                sender = "test@example.com"
            
            email_data = {
                'id': str(hash(user_input)),
                'text': user_input,
                'sender': sender
            }
            
            analysis = esis.predict(email_data)
            print_analysis(analysis, user_input, sender)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print_header("👋 Demo Complete!")
    print("Thank you for trying the Email Security Intelligence System!")
    print("\n🚀 Next steps:")
    print("• Run 'python esis_webapp.py' for the web interface")
    print("• Run 'python train_esis.py' for comprehensive training")
    print("• Check 'README_ESIS.md' for full documentation")

def main():
    """Main demo function"""
    try:
        demonstrate_esis()
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
