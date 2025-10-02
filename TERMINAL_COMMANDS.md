# Terminal Commands - Email Spam Classifier

## Quick Setup (2 minutes)

### 1. Install Required Packages
```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Train the Model
```bash
python quick_start.py
```
This will:
- Load the spam.csv dataset
- Train 3 ML models (Naive Bayes, Logistic Regression, Random Forest)
- Create an ensemble model
- Test with sample emails
- Save the model as 'quick_spam_model.pkl'

### 3. Use the Model
```bash
python use_model.py
```
This opens an interactive session where you can type emails to classify.

## Alternative: Batch Processing

### Classify Multiple Emails from CSV
```bash
python batch_classify.py
```
- Input: CSV file with email column
- Output: CSV file with classification results

## Expected Output

When you run `python quick_start.py`, you should see:
```
Quick Email Spam Classifier
========================================
Loading data...
Dataset: 5572 emails
Spam: 747 (13.4%)
Training spam classifier...
naive_bayes: 0.9193
logistic_regression: 0.9570
random_forest: 0.9794
Ensemble: 0.9570

========================================
Testing with sample emails:
========================================

Email: Free entry in 2 a wkly comp to win FA Cup final tk...
Result: SPAM (Prob: 0.791)

Email: Hey, how are you doing? Let's meet for coffee tomo...
Result: HAM (Prob: 0.006)

Quick setup complete!
Model saved as 'quick_spam_model.pkl'
```

## Files Created
- `quick_spam_model.pkl` - Trained model
- `quick_start.py` - Main training script
- `use_model.py` - Interactive classifier
- `batch_classify.py` - Batch processing

## That's it! Your spam classifier is ready to use.
