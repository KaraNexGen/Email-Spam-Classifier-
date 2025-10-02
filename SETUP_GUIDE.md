# 🚀 Email Spam Classifier - Complete Setup Guide

## Step-by-Step Terminal Instructions

### 1. Install Required Packages
```bash
# Install all required packages
pip install -r requirements.txt

# If you get permission errors, use:
pip install --user -r requirements.txt

# Or create a virtual environment (recommended):
python -m venv spam_classifier_env
spam_classifier_env\Scripts\activate  # On Windows
# source spam_classifier_env/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

### 2. Download NLTK Data (Required)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

### 3. Run the Demo (Easiest Start)
```bash
python demo.py
```

### 4. Train and Evaluate Models
```bash
python train_and_evaluate.py
```

### 5. Launch Interactive Dashboard
```bash
streamlit run visualization_dashboard.py
```

### 6. Start API Server
```bash
python api_server.py
```

### 7. Run Basic Classifier
```bash
python email_spam_classifier.py
```

## Quick Start (All-in-One)
```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"

# 3. Run demo
python demo.py
```

## Troubleshooting

### Common Issues:
1. **ModuleNotFoundError**: Install missing packages with `pip install package_name`
2. **NLTK Data Error**: Run the NLTK download command
3. **Permission Errors**: Use `--user` flag or virtual environment
4. **File Not Found**: Ensure `spam.csv` is in the same directory

### Virtual Environment (Recommended):
```bash
# Create virtual environment
python -m venv spam_env

# Activate (Windows)
spam_env\Scripts\activate

# Activate (Linux/Mac)
source spam_env/bin/activate

# Install packages
pip install -r requirements.txt

# Run demo
python demo.py
```
