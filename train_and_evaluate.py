"""
Training and Evaluation Script for Email Spam Classifier
========================================================

This script provides comprehensive training, evaluation, and analysis of the spam classifier:
- Advanced model training with hyperparameter tuning
- Cross-validation and performance metrics
- Feature importance analysis
- Model comparison and selection
- Comprehensive evaluation reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
from datetime import datetime
import warnings
from email_spam_classifier import AdvancedSpamClassifier, InnovativeFeatureExtractor

warnings.filterwarnings('ignore')

class ModelTrainer:
    """Advanced model trainer with comprehensive evaluation"""
    
    def __init__(self):
        self.classifier = AdvancedSpamClassifier()
        self.results = {}
        self.best_model = None
        self.feature_importance = None
        
    def load_and_prepare_data(self, filepath: str = 'spam.csv'):
        """Load and prepare data for training"""
        print("Loading dataset...")
        df = pd.read_csv(filepath, encoding='latin-1')
        df = df[['v1', 'v2']].dropna()
        df.columns = ['label', 'text']
        
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:")
        print(df['label'].value_counts())
        
        # Prepare features
        X, y, feature_names = self.classifier.prepare_data(df)
        
        return X, y, feature_names, df
    
    def train_with_hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray):
        """Train models with hyperparameter tuning"""
        print("\nTraining models with hyperparameter tuning...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with hyperparameter grids
        models_config = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
            }
        }
        
        # Train and tune each model
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train_balanced)
            
            # Evaluate best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['best_params'] = grid_search.best_params_
            metrics['best_score'] = grid_search.best_score_
            
            self.results[name] = {
                'model': best_model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Best CV Score: {grid_search.best_score_:.4f}")
            print(f"{name} - Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"{name} - Test F1-Score: {metrics['f1']:.4f}")
        
        # Train ensemble model
        self._train_ensemble_model(X_train_scaled, y_train_balanced, X_test_scaled, y_test)
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.scaler = scaler
        
        return X_test_scaled, y_test
    
    def _train_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_test: np.ndarray, y_test: np.ndarray):
        """Train ensemble model"""
        print("\nTraining ensemble model...")
        
        from sklearn.ensemble import VotingClassifier
        
        # Get best models
        best_models = []
        for name, result in self.results.items():
            best_models.append((name, result['model']))
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=best_models,
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        self.results['ensemble'] = {
            'model': ensemble,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Ensemble - Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Ensemble - Test F1-Score: {metrics['f1']:.4f}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
        """Calculate comprehensive metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba)
        }
    
    def cross_validate_models(self, X: np.ndarray, y: np.ndarray):
        """Perform cross-validation on all models"""
        print("\nPerforming cross-validation...")
        
        # Define models for CV
        models = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"Cross-validating {name}...")
            
            # Perform cross-validation
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            
            cv_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            
            print(f"{name} - CV F1-Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def analyze_feature_importance(self, feature_names: list):
        """Analyze feature importance"""
        print("\nAnalyzing feature importance...")
        
        # Get feature importance from Random Forest
        if 'random_forest' in self.results:
            rf_model = self.results['random_forest']['model']
            importance = rf_model.feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = feature_importance_df
            
            print("Top 20 Most Important Features:")
            print(feature_importance_df.head(20))
            
            return feature_importance_df
        
        return None
    
    def create_evaluation_plots(self, save_plots: bool = True):
        """Create comprehensive evaluation plots"""
        print("\nCreating evaluation plots...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Performance Comparison
        plt.subplot(3, 4, 1)
        models = list(self.results.keys())
        accuracies = [self.results[model]['metrics']['accuracy'] for model in models]
        f1_scores = [self.results[model]['metrics']['f1'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. ROC Curves
        plt.subplot(3, 4, 2)
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            auc = result['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        plt.subplot(3, 4, 3)
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['probabilities'])
            ap = result['metrics']['average_precision']
            plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Confusion Matrix for Best Model
        plt.subplot(3, 4, 4)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['f1'])
        cm = confusion_matrix(self.y_test, self.results[best_model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 5. Feature Importance (if available)
        if self.feature_importance is not None:
            plt.subplot(3, 4, 5)
            top_features = self.feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 15 Feature Importance')
            plt.gca().invert_yaxis()
        
        # 6. Model Metrics Heatmap
        plt.subplot(3, 4, 6)
        metrics_df = pd.DataFrame({
            model: result['metrics'] for model, result in self.results.items()
        }).T
        metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Model Metrics Heatmap')
        plt.ylabel('Models')
        
        # 7. Prediction Probability Distribution
        plt.subplot(3, 4, 7)
        for name, result in self.results.items():
            plt.hist(result['probabilities'], alpha=0.5, label=name, bins=20)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Class Distribution
        plt.subplot(3, 4, 8)
        class_counts = np.bincount(self.y_test)
        plt.pie(class_counts, labels=['Ham', 'Spam'], autopct='%1.1f%%', startangle=90)
        plt.title('Test Set Class Distribution')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
            print("Evaluation plots saved as 'model_evaluation_plots.png'")
        
        plt.show()
    
    def generate_report(self, save_report: bool = True):
        """Generate comprehensive evaluation report"""
        print("\nGenerating evaluation report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.y_test),
                'spam_samples': int(np.sum(self.y_test)),
                'ham_samples': int(len(self.y_test) - np.sum(self.y_test)),
                'spam_ratio': float(np.mean(self.y_test))
            },
            'model_results': {}
        }
        
        # Add model results
        for name, result in self.results.items():
            report['model_results'][name] = {
                'metrics': result['metrics'],
                'best_params': result.get('best_params', None),
                'best_cv_score': result.get('best_score', None)
            }
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['f1'])
        report['best_model'] = {
            'name': best_model,
            'f1_score': self.results[best_model]['metrics']['f1'],
            'accuracy': self.results[best_model]['metrics']['accuracy']
        }
        
        # Add feature importance if available
        if self.feature_importance is not None:
            report['top_features'] = self.feature_importance.head(20).to_dict('records')
        
        if save_report:
            with open('evaluation_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            print("Evaluation report saved as 'evaluation_report.json'")
        
        return report
    
    def save_best_model(self, filepath: str = 'best_spam_classifier.pkl'):
        """Save the best performing model"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['f1'])
        best_model = self.results[best_model_name]['model']
        
        model_data = {
            'model': best_model,
            'model_name': best_model_name,
            'scaler': self.scaler,
            'metrics': self.results[best_model_name]['metrics'],
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Best model ({best_model_name}) saved to {filepath}")
        
        return best_model_name, self.results[best_model_name]['metrics']


def main():
    """Main function to run training and evaluation"""
    print("Advanced Email Spam Classifier - Training and Evaluation")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load and prepare data
    X, y, feature_names, df = trainer.load_and_prepare_data()
    
    # Perform cross-validation
    cv_results = trainer.cross_validate_models(X, y)
    
    # Train models with hyperparameter tuning
    X_test, y_test = trainer.train_with_hyperparameter_tuning(X, y)
    
    # Analyze feature importance
    feature_importance = trainer.analyze_feature_importance(feature_names)
    
    # Create evaluation plots
    trainer.create_evaluation_plots()
    
    # Generate comprehensive report
    report = trainer.generate_report()
    
    # Save best model
    best_model_name, best_metrics = trainer.save_best_model()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Best Model: {best_model_name}")
    print(f"Best F1-Score: {best_metrics['f1']:.4f}")
    print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print("\nAll Model Results:")
    for name, result in trainer.results.items():
        print(f"{name:20} - F1: {result['metrics']['f1']:.4f}, Acc: {result['metrics']['accuracy']:.4f}")
    
    print(f"\nFiles generated:")
    print("- model_evaluation_plots.png")
    print("- evaluation_report.json")
    print("- best_spam_classifier.pkl")


if __name__ == "__main__":
    main()
