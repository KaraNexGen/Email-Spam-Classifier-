"""
FastAPI Server for Email Spam Classifier
========================================

This module provides a REST API for real-time spam detection with:
- Single email classification
- Batch email processing
- Model performance metrics
- Health check endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from email_spam_classifier import AdvancedSpamClassifier
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Email Spam Classifier API",
    description="Advanced ML-based email spam detection API with innovative features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None

# Pydantic models for request/response
class EmailRequest(BaseModel):
    text: str = Field(..., description="Email text content", min_length=1, max_length=10000)
    include_features: bool = Field(False, description="Include feature analysis in response")

class BatchEmailRequest(BaseModel):
    emails: List[EmailRequest] = Field(..., description="List of emails to classify", max_items=100)
    include_features: bool = Field(False, description="Include feature analysis in response")

class EmailResponse(BaseModel):
    text: str
    spam_probability: float = Field(..., ge=0, le=1, description="Probability of being spam")
    classification: str = Field(..., description="Classification result: SPAM or HAM")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the prediction")
    features: Optional[Dict] = Field(None, description="Feature analysis (if requested)")
    timestamp: datetime

class BatchEmailResponse(BaseModel):
    results: List[EmailResponse]
    total_processed: int
    spam_count: int
    ham_count: int
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime
    version: str

class ModelInfoResponse(BaseModel):
    model_name: str
    algorithms: List[str]
    features_count: int
    training_date: Optional[str]
    accuracy: Optional[float]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load the trained model on startup"""
    global classifier
    try:
        classifier = AdvancedSpamClassifier()
        if os.path.exists('spam_classifier_model.pkl'):
            classifier.load_model('spam_classifier_model.pkl')
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found. Please train the model first.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if classifier and classifier.is_trained else "unhealthy",
        model_loaded=classifier is not None and classifier.is_trained,
        timestamp=datetime.now(),
        version="1.0.0"
    )

# Model information endpoint
@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if not classifier or not classifier.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name="Advanced Email Spam Classifier",
        algorithms=list(classifier.models.keys()),
        features_count=len(classifier.feature_extractor.extract_all_features(["sample"]).columns),
        training_date=None,  # Could be added to model metadata
        accuracy=None  # Could be added to model metadata
    )

# Single email classification endpoint
@app.post("/classify", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    """Classify a single email as spam or ham"""
    if not classifier or not classifier.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        predictions = classifier.predict([request.text])
        ensemble_prob = predictions['ensemble'][0]
        
        # Determine classification and confidence
        classification = "SPAM" if ensemble_prob > 0.5 else "HAM"
        confidence = max(ensemble_prob, 1 - ensemble_prob)
        
        # Extract features if requested
        features = None
        if request.include_features:
            features_df = classifier.feature_extractor.extract_all_features([request.text])
            features = features_df.iloc[0].to_dict()
        
        return EmailResponse(
            text=request.text,
            spam_probability=ensemble_prob,
            classification=classification,
            confidence=confidence,
            features=features,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error classifying email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

# Batch email classification endpoint
@app.post("/classify/batch", response_model=BatchEmailResponse)
async def classify_batch_emails(request: BatchEmailRequest):
    """Classify multiple emails in batch"""
    if not classifier or not classifier.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Extract texts
        texts = [email.text for email in request.emails]
        
        # Make predictions
        predictions = classifier.predict(texts)
        ensemble_probs = predictions['ensemble']
        
        # Process results
        results = []
        spam_count = 0
        ham_count = 0
        
        for i, (email, prob) in enumerate(zip(request.emails, ensemble_probs)):
            classification = "SPAM" if prob > 0.5 else "HAM"
            confidence = max(prob, 1 - prob)
            
            if classification == "SPAM":
                spam_count += 1
            else:
                ham_count += 1
            
            # Extract features if requested
            features = None
            if request.include_features:
                features_df = classifier.feature_extractor.extract_all_features([email.text])
                features = features_df.iloc[0].to_dict()
            
            results.append(EmailResponse(
                text=email.text,
                spam_probability=prob,
                classification=classification,
                confidence=confidence,
                features=features,
                timestamp=datetime.now()
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchEmailResponse(
            results=results,
            total_processed=len(results),
            spam_count=spam_count,
            ham_count=ham_count,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error in batch classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch classification error: {str(e)}")

# Get model predictions for comparison
@app.post("/classify/detailed")
async def get_detailed_predictions(request: EmailRequest):
    """Get detailed predictions from all models"""
    if not classifier or not classifier.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = classifier.predict([request.text])
        
        # Format detailed results
        detailed_results = {
            "text": request.text,
            "timestamp": datetime.now(),
            "predictions": {}
        }
        
        for model_name, pred in predictions.items():
            detailed_results["predictions"][model_name] = {
                "spam_probability": float(pred[0]),
                "classification": "SPAM" if pred[0] > 0.5 else "HAM",
                "confidence": float(max(pred[0], 1 - pred[0]))
            }
        
        # Add ensemble result
        ensemble_prob = predictions['ensemble'][0]
        detailed_results["ensemble"] = {
            "spam_probability": float(ensemble_prob),
            "classification": "SPAM" if ensemble_prob > 0.5 else "HAM",
            "confidence": float(max(ensemble_prob, 1 - ensemble_prob))
        }
        
        return detailed_results
    
    except Exception as e:
        logger.error(f"Error getting detailed predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detailed prediction error: {str(e)}")

# Feature analysis endpoint
@app.post("/analyze/features")
async def analyze_features(request: EmailRequest):
    """Analyze features of an email text"""
    if not classifier or not classifier.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract features
        features_df = classifier.feature_extractor.extract_all_features([request.text])
        features = features_df.iloc[0].to_dict()
        
        # Get top features
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = dict(sorted_features[:20])
        
        return {
            "text": request.text,
            "all_features": features,
            "top_features": top_features,
            "feature_count": len(features),
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Error analyzing features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature analysis error: {str(e)}")

# Model retraining endpoint (for future use)
@app.post("/model/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the model with new data (placeholder for future implementation)"""
    # This would be implemented to retrain the model with new data
    # For now, it's a placeholder
    return {
        "message": "Model retraining initiated",
        "status": "in_progress",
        "timestamp": datetime.now()
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Email Spam Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "classify": "/classify",
            "batch_classify": "/classify/batch",
            "detailed_predictions": "/classify/detailed",
            "feature_analysis": "/analyze/features",
            "model_info": "/model/info"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
