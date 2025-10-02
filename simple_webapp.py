"""
Simple ESIS Web Application
==========================
A simplified web application for ESIS that works without complex dependencies
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import uuid
from esis_simple import SimpleESIS, create_demo_dataset

app = Flask(__name__)
app.secret_key = 'esis_simple_secret_key_2024'
CORS(app)

# Global ESIS instance
esis = None
analysis_history = []

def init_esis():
    """Initialize ESIS system"""
    global esis
    esis = SimpleESIS()
    
    # Try to load existing model
    if os.path.exists('esis_simple_model.pkl'):
        try:
            esis.load_model('esis_simple_model.pkl')
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
    
    # Create demo dataset
    df = create_demo_dataset()
    
    # Train the model
    X, y, feature_names = esis.prepare_data(df)
    esis.train_models(X, y)
    esis.save_model('esis_simple_model.pkl')
    print("✅ ESIS demo model trained and saved")

@app.route('/')
def index():
    """Main dashboard"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ESIS - Email Security Intelligence System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            .threat-critical { color: #dc3545; font-weight: bold; }
            .threat-high { color: #fd7e14; font-weight: bold; }
            .threat-medium { color: #ffc107; font-weight: bold; }
            .threat-low { color: #28a745; font-weight: bold; }
            
            .classification-spam { background-color: #f8d7da; border-left: 4px solid #dc3545; }
            .classification-phishing { background-color: #fff3cd; border-left: 4px solid #ffc107; }
            .classification-ham { background-color: #d4edda; border-left: 4px solid #28a745; }
            
            .analysis-card {
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            
            .risk-factor {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 5px 10px;
                margin: 2px;
                display: inline-block;
                font-size: 0.9em;
            }
            
            .confidence-bar {
                height: 20px;
                border-radius: 10px;
                background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%);
                position: relative;
            }
            
            .confidence-indicator {
                position: absolute;
                top: 0;
                height: 100%;
                width: 3px;
                background-color: #000;
                border-radius: 2px;
            }
        </style>
    </head>
    <body class="bg-light">
        <!-- Navigation -->
        <nav class="navbar navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="#">
                    <i class="fas fa-shield-alt"></i> ESIS
                </a>
                <span class="navbar-text">
                    Email Security Intelligence System
                </span>
            </div>
        </nav>

        <div class="container mt-4">
            <!-- Header -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card analysis-card">
                        <div class="card-body text-center">
                            <h1 class="card-title">
                                <i class="fas fa-shield-alt text-primary"></i>
                                Email Security Intelligence System
                            </h1>
                            <p class="card-text text-muted">
                                Advanced multi-threat email analysis with explainable AI
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Analysis Interface -->
            <div class="row">
                <!-- Email Analysis Form -->
                <div class="col-lg-6">
                    <div class="card analysis-card">
                        <div class="card-header">
                            <h5><i class="fas fa-search"></i> Email Threat Analysis</h5>
                        </div>
                        <div class="card-body">
                            <form id="emailForm">
                                <div class="mb-3">
                                    <label for="senderEmail" class="form-label">Sender Email</label>
                                    <input type="email" class="form-control" id="senderEmail" 
                                           placeholder="sender@example.com" value="security@bank-verification.com">
                                </div>
                                <div class="mb-3">
                                    <label for="emailText" class="form-label">Email Content</label>
                                    <textarea class="form-control" id="emailText" rows="8" 
                                              placeholder="Paste email content here...">URGENT: Your account has been suspended. Click here to verify: bit.ly/suspicious-link</textarea>
                                </div>
                                <button type="submit" class="btn btn-primary w-100">
                                    <i class="fas fa-search"></i> Analyze Email
                                </button>
                            </form>
                            
                            <!-- Loading Indicator -->
                            <div id="loading" class="text-center mt-3" style="display: none;">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Analyzing...</span>
                                </div>
                                <p class="mt-2">Analyzing email for threats...</p>
                            </div>
                        </div>
                    </div>

                    <!-- Quick Examples -->
                    <div class="card analysis-card mt-3">
                        <div class="card-header">
                            <h5><i class="fas fa-lightbulb"></i> Quick Examples</h5>
                        </div>
                        <div class="card-body">
                            <div class="d-grid gap-2">
                                <button class="btn btn-outline-primary btn-sm" onclick="loadExample('phishing')">
                                    <i class="fas fa-exclamation-triangle"></i> Phishing Example
                                </button>
                                <button class="btn btn-outline-warning btn-sm" onclick="loadExample('spam')">
                                    <i class="fas fa-spam"></i> Spam Example
                                </button>
                                <button class="btn btn-outline-success btn-sm" onclick="loadExample('ham')">
                                    <i class="fas fa-check-circle"></i> Legitimate Example
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Analysis Results -->
                <div class="col-lg-6">
                    <div id="analysisResults" style="display: none;">
                        <div class="card analysis-card">
                            <div class="card-header">
                                <h5><i class="fas fa-chart-line"></i> Analysis Results</h5>
                            </div>
                            <div class="card-body">
                                <div id="analysisContent"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Statistics -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card analysis-card">
                        <div class="card-header">
                            <h5><i class="fas fa-chart-bar"></i> System Statistics</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="text-center">
                                        <h3 id="totalAnalyses" class="text-primary">0</h3>
                                        <p class="text-muted">Total Analyses</p>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="text-center">
                                        <h3 id="avgConfidence" class="text-success">0%</h3>
                                        <p class="text-muted">Avg Confidence</p>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="text-center">
                                        <h3 id="avgReputation" class="text-info">0%</h3>
                                        <p class="text-muted">Avg Sender Rep</p>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="text-center">
                                        <h3 id="threatsDetected" class="text-danger">0</h3>
                                        <p class="text-muted">Threats Detected</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Analysis History -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card analysis-card">
                        <div class="card-header">
                            <h5><i class="fas fa-history"></i> Recent Analyses</h5>
                        </div>
                        <div class="card-body">
                            <div id="analysisHistory"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Load examples
            const examples = {
                phishing: {
                    sender: "security@bank-verification.com",
                    text: "URGENT: Your account has been suspended due to suspicious activity. Click here to verify your identity immediately: bit.ly/suspicious-link. Failure to verify within 24 hours will result in permanent account closure."
                },
                spam: {
                    sender: "winner@lottery-prize.org",
                    text: "CONGRATULATIONS! You have been selected to receive $10,000! This is a limited time offer. Click here to claim your prize now: bit.ly/fake-prize. Act fast - offer expires in 24 hours!"
                },
                ham: {
                    sender: "john@gmail.com",
                    text: "Hi Sarah, I hope you're doing well. I wanted to confirm our meeting tomorrow at 2 PM in the conference room. I'll bring the quarterly reports we discussed. Let me know if you need to reschedule. Thanks!"
                }
            };

            function loadExample(type) {
                const example = examples[type];
                document.getElementById('senderEmail').value = example.sender;
                document.getElementById('emailText').value = example.text;
            }

            // Email analysis form
            document.getElementById('emailForm').addEventListener('submit', function(e) {
                e.preventDefault();
                analyzeEmail();
            });

            function analyzeEmail() {
                const emailData = {
                    text: document.getElementById('emailText').value,
                    sender: document.getElementById('senderEmail').value
                };

                if (!emailData.text.trim()) {
                    alert('Please enter email content');
                    return;
                }

                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('analysisResults').style.display = 'none';

                fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(emailData)
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    
                    if (data.success) {
                        displayAnalysis(data.analysis);
                        loadStats();
                        loadHistory();
                    } else {
                        alert('Analysis failed: ' + data.error);
                    }
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    alert('Error: ' + error.message);
                });
            }

            function displayAnalysis(analysis) {
                const content = `
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Classification</h6>
                            <div class="classification-${analysis.classification} p-3 rounded">
                                <strong>${analysis.classification.toUpperCase()}</strong>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h6>Threat Level</h6>
                            <div class="threat-${analysis.threat_level} p-3 rounded">
                                <strong>${analysis.threat_level.toUpperCase()}</strong>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-3">
                        <div class="col-md-6">
                            <h6>Confidence</h6>
                            <div class="confidence-bar">
                                <div class="confidence-indicator" style="left: ${analysis.confidence * 100}%"></div>
                            </div>
                            <small class="text-muted">${(analysis.confidence * 100).toFixed(1)}%</small>
                        </div>
                        <div class="col-md-6">
                            <h6>Sender Reputation</h6>
                            <div class="progress">
                                <div class="progress-bar" style="width: ${analysis.sender_reputation * 100}%"></div>
                            </div>
                            <small class="text-muted">${(analysis.sender_reputation * 100).toFixed(1)}%</small>
                        </div>
                    </div>
                    
                    <div class="mt-3">
                        <h6>Risk Factors</h6>
                        <div>
                            ${analysis.risk_factors.map(factor => 
                                `<span class="risk-factor">${factor}</span>`
                            ).join('')}
                        </div>
                    </div>
                `;
                
                document.getElementById('analysisContent').innerHTML = content;
                document.getElementById('analysisResults').style.display = 'block';
            }

            function loadStats() {
                fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const stats = data.stats;
                        document.getElementById('totalAnalyses').textContent = stats.total_analyses;
                        document.getElementById('avgConfidence').textContent = (stats.avg_confidence * 100).toFixed(1) + '%';
                        document.getElementById('avgReputation').textContent = (stats.avg_sender_reputation * 100).toFixed(1) + '%';
                        
                        const threats = (stats.classifications.phishing || 0) + (stats.classifications.spam || 0);
                        document.getElementById('threatsDetected').textContent = threats;
                    }
                });
            }

            function loadHistory() {
                fetch('/history')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const history = data.history.slice(-10).reverse();
                        const content = history.map(item => `
                            <div class="card mb-2">
                                <div class="card-body p-2">
                                    <div class="row">
                                        <div class="col-md-2">
                                            <strong>${item.classification.toUpperCase()}</strong>
                                        </div>
                                        <div class="col-md-2">
                                            <span class="threat-${item.threat_level}">${item.threat_level.toUpperCase()}</span>
                                        </div>
                                        <div class="col-md-4">
                                            <small>${item.text}</small>
                                        </div>
                                        <div class="col-md-2">
                                            <small>${(item.confidence * 100).toFixed(1)}%</small>
                                        </div>
                                        <div class="col-md-2">
                                            <small>${new Date(item.timestamp).toLocaleTimeString()}</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `).join('');
                        
                        document.getElementById('analysisHistory').innerHTML = content || '<p class="text-muted">No analyses yet</p>';
                    }
                });
            }

            // Load initial data
            loadStats();
            loadHistory();
        </script>
    </body>
    </html>
    '''

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
            'sender': data.get('sender', 'unknown@example.com')
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

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'esis_loaded': esis is not None and esis.is_trained,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-simple'
    })

if __name__ == '__main__':
    print("🛡️ Starting Email Security Intelligence System (ESIS) - Simple Version")
    print("=" * 70)
    
    # Initialize ESIS
    init_esis()
    
    print("✅ ESIS Web Application ready!")
    print("🌐 Access the application at: http://localhost:5000")
    print("📊 Dashboard: http://localhost:5000")
    print("🔍 API Documentation: http://localhost:5000/health")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
