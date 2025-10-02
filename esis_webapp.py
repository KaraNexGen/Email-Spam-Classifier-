"""
ESIS Web Application
===================
Professional Flask web application for Email Security Intelligence System
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import uuid
from esis_core import ESISCore, EmailAnalysis
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)
app.secret_key = 'esis_secret_key_2024'
CORS(app)

# Global ESIS instance
esis = None
analysis_history = []

def init_esis():
    """Initialize ESIS system"""
    global esis
    esis = ESISCore()
    
    # Try to load existing model
    if os.path.exists('esis_model.pkl'):
        try:
            esis.load_model('esis_model.pkl')
            print("✅ ESIS model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load existing model: {e}")
            print("Training new model...")
            train_demo_model()
    else:
        print("Training new ESIS model...")
        train_demo_model()

def train_demo_model():
    """Train ESIS with demo data"""
    global esis
    
    # Create comprehensive demo dataset
    demo_data = pd.DataFrame({
        'text': [
            # Phishing emails
            "URGENT: Your account has been suspended. Click here to verify: bit.ly/suspicious-link",
            "Your bank account needs immediate verification. Please click the link to avoid closure.",
            "Security Alert: Unusual activity detected. Verify your identity now.",
            "Your PayPal account has been limited. Click here to restore access immediately.",
            "Amazon: Your order cannot be processed. Update payment information now.",
            "Microsoft: Your account will be closed. Verify your identity to prevent closure.",
            "Apple ID: Suspicious login detected. Secure your account now.",
            "Netflix: Payment failed. Update your billing information immediately.",
            
            # Spam emails
            "Free money! Win $1000 now! Limited time offer!",
            "Congratulations! You've won a free iPhone! Claim your prize now!",
            "URGENT: You have won a prize! Call now to claim your reward!",
            "Free entry in 2 a wkly comp to win FA Cup final tkts!",
            "SIX chances to win CASH! From 100 to 20,000 pounds!",
            "WINNER!! As a valued network customer you have been selected!",
            "FreeMsg Hey there darling it's been 3 week's now and no word back!",
            "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles!",
            
            # Ham emails
            "Hey, how are you doing? Let's meet for coffee tomorrow.",
            "Thanks for the meeting yesterday. I'll send the report by Friday.",
            "Can we schedule a call for tomorrow morning?",
            "I'll see you at the office tomorrow morning.",
            "Thanks for your help with the project. Great work!",
            "The presentation went well. Thanks for your support.",
            "I've been searching for the right words to thank you for this breather.",
            "I HAVE A DATE ON SUNDAY WITH WILL!!",
            "Even my brother is not like to speak with me. They treat me like aids patent.",
            "As per your request 'Melle Melle' has been set as your callertune.",
            "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight.",
            "Oh k...i'm watching here:)",
            "Fine if that's the way u feel. That's the way its gota b"
        ],
        'sender': [
            # Phishing senders
            "security@bank-verification.com",
            "noreply@yourbank.com",
            "alerts@security-center.org",
            "support@paypal-verification.net",
            "orders@amazon-security.com",
            "account@microsoft-verification.org",
            "security@apple-id-verification.com",
            "billing@netflix-security.org",
            
            # Spam senders
            "winner@lottery-prize.org",
            "prizes@free-iphone.com",
            "rewards@prize-claim.net",
            "competitions@free-tickets.org",
            "cash@win-money.com",
            "network@customer-rewards.org",
            "dating@free-messages.com",
            "mobile@upgrade-offers.net",
            
            # Ham senders
            "john@gmail.com",
            "sarah@company.com",
            "mike@business.org",
            "lisa@office.com",
            "team@project.net",
            "manager@company.com",
            "colleague@work.org",
            "friend@personal.com",
            "family@home.org",
            "brother@family.com",
            "sister@personal.org",
            "friend@social.com",
            "buddy@friends.net"
        ],
        'label': [
            'phishing', 'phishing', 'phishing', 'phishing', 'phishing', 'phishing', 'phishing', 'phishing',
            'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam',
            'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham'
        ]
    })
    
    # Train the model
    X, y, feature_names = esis.prepare_data(demo_data)
    esis.train_models(X, y)
    esis.save_model('esis_model.pkl')
    print("✅ ESIS demo model trained and saved")

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_email():
    """Analyze email for threats"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Email text is required'}), 400
        
        # Prepare email data
        email_data = {
            'id': str(uuid.uuid4()),
            'text': data['text'],
            'sender': data.get('sender', 'unknown@example.com'),
            'subject': data.get('subject', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # Analyze with ESIS
        analysis = esis.predict(email_data)
        
        # Store in history
        analysis_record = {
            'id': analysis.email_id,
            'text': email_data['text'][:100] + '...' if len(email_data['text']) > 100 else email_data['text'],
            'sender': email_data['sender'],
            'classification': analysis.classification,
            'threat_level': analysis.threat_level,
            'confidence': analysis.confidence,
            'sender_reputation': analysis.sender_reputation,
            'risk_factors': analysis.risk_factors,
            'timestamp': analysis.timestamp.isoformat()
        }
        
        analysis_history.append(analysis_record)
        
        # Keep only last 100 analyses
        if len(analysis_history) > 100:
            analysis_history.pop(0)
        
        # Prepare response
        response = {
            'success': True,
            'analysis': {
                'email_id': analysis.email_id,
                'classification': analysis.classification,
                'threat_level': analysis.threat_level,
                'confidence': round(analysis.confidence, 3),
                'sender_reputation': round(analysis.sender_reputation, 3),
                'risk_factors': analysis.risk_factors,
                'explanations': analysis.explanations,
                'timestamp': analysis.timestamp.isoformat()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Batch analyze multiple emails"""
    try:
        data = request.get_json()
        
        if not data or 'emails' not in data:
            return jsonify({'error': 'Emails array is required'}), 400
        
        results = []
        
        for email_data in data['emails']:
            try:
                # Prepare email data
                email_info = {
                    'id': str(uuid.uuid4()),
                    'text': email_data.get('text', ''),
                    'sender': email_data.get('sender', 'unknown@example.com'),
                    'subject': email_data.get('subject', ''),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Analyze with ESIS
                analysis = esis.predict(email_info)
                
                results.append({
                    'email_id': analysis.email_id,
                    'classification': analysis.classification,
                    'threat_level': analysis.threat_level,
                    'confidence': round(analysis.confidence, 3),
                    'sender_reputation': round(analysis.sender_reputation, 3),
                    'risk_factors': analysis.risk_factors[:3],  # Limit to top 3
                    'timestamp': analysis.timestamp.isoformat()
                })
                
            except Exception as e:
                results.append({
                    'error': f'Failed to analyze email: {str(e)}',
                    'email_id': str(uuid.uuid4())
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}), 500

@app.route('/history')
def get_history():
    """Get analysis history"""
    return jsonify({
        'success': True,
        'history': analysis_history[-50:]  # Last 50 analyses
    })

@app.route('/stats')
def get_stats():
    """Get system statistics"""
    if not analysis_history:
        return jsonify({
            'success': True,
            'stats': {
                'total_analyses': 0,
                'classifications': {'ham': 0, 'spam': 0, 'phishing': 0},
                'threat_levels': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                'avg_confidence': 0,
                'avg_sender_reputation': 0
            }
        })
    
    # Calculate statistics
    total = len(analysis_history)
    classifications = {}
    threat_levels = {}
    confidences = []
    reputations = []
    
    for analysis in analysis_history:
        # Classifications
        cls = analysis['classification']
        classifications[cls] = classifications.get(cls, 0) + 1
        
        # Threat levels
        threat = analysis['threat_level']
        threat_levels[threat] = threat_levels.get(threat, 0) + 1
        
        # Confidences and reputations
        confidences.append(analysis['confidence'])
        reputations.append(analysis['sender_reputation'])
    
    return jsonify({
        'success': True,
        'stats': {
            'total_analyses': total,
            'classifications': classifications,
            'threat_levels': threat_levels,
            'avg_confidence': round(np.mean(confidences), 3),
            'avg_sender_reputation': round(np.mean(reputations), 3)
        }
    })

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for adaptive learning"""
    try:
        data = request.get_json()
        
        if not data or 'email_id' not in data or 'feedback' not in data:
            return jsonify({'error': 'Email ID and feedback are required'}), 400
        
        # Store feedback for future retraining
        feedback_record = {
            'email_id': data['email_id'],
            'user_feedback': data['feedback'],
            'timestamp': datetime.now().isoformat()
        }
        
        # In a real system, this would be stored in a database
        # and used for online learning
        print(f"Feedback received: {feedback_record}")
        
        return jsonify({
            'success': True,
            'message': 'Feedback received. Thank you for helping improve the system!'
        })
    
    except Exception as e:
        return jsonify({'error': f'Feedback submission failed: {str(e)}'}), 500

@app.route('/visualizations')
def get_visualizations():
    """Get visualization data"""
    if not analysis_history:
        return jsonify({'success': True, 'charts': {}})
    
    # Prepare data for visualizations
    df = pd.DataFrame(analysis_history)
    
    # Classification distribution
    class_counts = df['classification'].value_counts()
    class_chart = {
        'data': [{'x': list(class_counts.index), 'y': list(class_counts.values)}],
        'layout': {'title': 'Email Classifications', 'xaxis': {'title': 'Classification'}, 'yaxis': {'title': 'Count'}}
    }
    
    # Threat level distribution
    threat_counts = df['threat_level'].value_counts()
    threat_chart = {
        'data': [{'x': list(threat_counts.index), 'y': list(threat_counts.values)}],
        'layout': {'title': 'Threat Levels', 'xaxis': {'title': 'Threat Level'}, 'yaxis': {'title': 'Count'}}
    }
    
    # Confidence over time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    confidence_chart = {
        'data': [{'x': df['timestamp'].dt.strftime('%H:%M'), 'y': df['confidence']}],
        'layout': {'title': 'Confidence Over Time', 'xaxis': {'title': 'Time'}, 'yaxis': {'title': 'Confidence'}}
    }
    
    return jsonify({
        'success': True,
        'charts': {
            'classifications': class_chart,
            'threat_levels': threat_chart,
            'confidence_trend': confidence_chart
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'esis_loaded': esis is not None and esis.is_trained,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🛡️ Starting Email Security Intelligence System (ESIS) Web Application")
    print("=" * 70)
    
    # Initialize ESIS
    init_esis()
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("✅ ESIS Web Application ready!")
    print("🌐 Access the application at: http://localhost:5000")
    print("📊 Dashboard: http://localhost:5000")
    print("🔍 API Documentation: http://localhost:5000/health")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
