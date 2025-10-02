"""
Batch Email Classification
========================
Classify multiple emails from a CSV file.
"""

import pandas as pd
from quick_start import QuickSpamClassifier

def classify_emails_from_csv(input_file, output_file):
    """Classify emails from CSV file and save results"""
    
    # Load the trained model
    classifier = QuickSpamClassifier()
    classifier.load_model('quick_spam_model.pkl')
    
    # Load input data
    print(f"Loading emails from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Assume the email text is in a column called 'text' or 'email'
    text_column = None
    for col in ['text', 'email', 'message', 'content']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        print("Error: No text column found. Expected columns: 'text', 'email', 'message', or 'content'")
        return
    
    print(f"Found {len(df)} emails to classify...")
    
    # Classify each email
    results = []
    for i, row in df.iterrows():
        email_text = str(row[text_column])
        result = classifier.predict(email_text)
        
        results.append({
            'original_text': email_text,
            'classification': result['classification'],
            'spam_probability': result['spam_probability'],
            'confidence': result['confidence']
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} emails...")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary
    spam_count = len(results_df[results_df['classification'] == 'SPAM'])
    ham_count = len(results_df[results_df['classification'] == 'HAM'])
    
    print(f"\nSummary:")
    print(f"Total emails: {len(results_df)}")
    print(f"Spam: {spam_count} ({spam_count/len(results_df)*100:.1f}%)")
    print(f"Ham: {ham_count} ({ham_count/len(results_df)*100:.1f}%)")

def main():
    """Main function"""
    print("Batch Email Classification")
    print("=" * 30)
    
    # Example usage
    input_file = input("Enter input CSV file path (or press Enter for 'emails.csv'): ").strip()
    if not input_file:
        input_file = 'emails.csv'
    
    output_file = input("Enter output CSV file path (or press Enter for 'results.csv'): ").strip()
    if not output_file:
        output_file = 'results.csv'
    
    try:
        classify_emails_from_csv(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        print("Make sure the file exists and contains email data.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
