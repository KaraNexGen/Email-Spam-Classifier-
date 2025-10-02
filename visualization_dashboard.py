"""
Interactive Visualization Dashboard for Email Spam Classifier
============================================================

This module provides comprehensive visualizations for:
- Dataset analysis
- Feature importance
- Model performance
- Real-time predictions
- Spam pattern analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import streamlit as st
from collections import Counter
import re
from email_spam_classifier import AdvancedSpamClassifier, InnovativeFeatureExtractor

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpamVisualizationDashboard:
    """Interactive dashboard for spam classifier analysis"""
    
    def __init__(self):
        self.feature_extractor = InnovativeFeatureExtractor()
        
    def create_dataset_overview(self, df: pd.DataFrame):
        """Create dataset overview visualizations"""
        st.subheader("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution
            fig = px.pie(
                df, 
                names='label', 
                title="Spam vs Ham Distribution",
                color_discrete_map={'spam': '#ff6b6b', 'ham': '#4ecdc4'}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Text length distribution
            df['text_length'] = df['text'].str.len()
            fig = px.histogram(
                df, 
                x='text_length', 
                color='label',
                title="Text Length Distribution",
                nbins=50,
                color_discrete_map={'spam': '#ff6b6b', 'ham': '#4ecdc4'}
            )
            fig.update_layout(xaxis_title="Text Length (characters)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    def create_word_clouds(self, df: pd.DataFrame):
        """Create word clouds for spam and ham messages"""
        st.subheader("☁️ Word Clouds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Spam Messages**")
            spam_texts = ' '.join(df[df['label'] == 'spam']['text'].fillna(''))
            if spam_texts:
                wordcloud = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='Reds',
                    max_words=100
                ).generate(spam_texts)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        with col2:
            st.write("**Ham Messages**")
            ham_texts = ' '.join(df[df['label'] == 'ham']['text'].fillna(''))
            if ham_texts:
                wordcloud = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='Blues',
                    max_words=100
                ).generate(ham_texts)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
    
    def analyze_spam_patterns(self, df: pd.DataFrame):
        """Analyze common spam patterns"""
        st.subheader("🔍 Spam Pattern Analysis")
        
        spam_df = df[df['label'] == 'spam']
        
        # Extract features for spam messages
        spam_features = self.feature_extractor.extract_all_features(spam_df['text'].tolist())
        
        # Create feature importance plot
        feature_importance = spam_features.mean().sort_values(ascending=False).head(15)
        
        fig = px.bar(
            x=feature_importance.values,
            y=feature_importance.index,
            orientation='h',
            title="Top Spam Indicators (Average Values)",
            color=feature_importance.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            xaxis_title="Average Value",
            yaxis_title="Features",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Spam keyword analysis
        st.write("**Common Spam Keywords**")
        spam_words = []
        for text in spam_df['text']:
            words = re.findall(r'\b\w+\b', text.lower())
            spam_words.extend(words)
        
        word_freq = Counter(spam_words)
        common_words = dict(word_freq.most_common(20))
        
        fig = px.bar(
            x=list(common_words.values()),
            y=list(common_words.keys()),
            orientation='h',
            title="Most Common Words in Spam Messages",
            color=list(common_words.values()),
            color_continuous_scale='Oranges'
        )
        fig.update_layout(
            xaxis_title="Frequency",
            yaxis_title="Words",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def create_feature_correlation_heatmap(self, df: pd.DataFrame):
        """Create feature correlation heatmap"""
        st.subheader("🔥 Feature Correlation Analysis")
        
        # Extract features for all messages
        features_df = self.feature_extractor.extract_all_features(df['text'].tolist())
        features_df['label'] = (df['label'] == 'spam').astype(int)
        
        # Select numeric features for correlation
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        correlation_matrix = numeric_features.corr()
        
        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_model_performance_dashboard(self, model_results: dict):
        """Create model performance comparison dashboard"""
        st.subheader("📈 Model Performance Comparison")
        
        models = list(model_results.keys())
        accuracies = [model_results[model]['accuracy'] for model in models]
        precisions = [model_results[model]['precision'] for model in models]
        recalls = [model_results[model]['recall'] for model in models]
        f1_scores = [model_results[model]['f1'] for model in models]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=precisions, name='Precision', marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=models, y=recalls, name='Recall', marker_color='lightcoral'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name='F1-Score', marker_color='lightyellow'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Model Performance Metrics"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_real_time_prediction_interface(self, classifier):
        """Create real-time prediction interface"""
        st.subheader("🚀 Real-time Spam Detection")
        
        # Text input
        user_text = st.text_area(
            "Enter email text to classify:",
            placeholder="Paste your email content here...",
            height=150
        )
        
        if st.button("Classify Email"):
            if user_text.strip():
                # Make prediction
                predictions = classifier.predict([user_text])
                ensemble_prob = predictions['ensemble'][0]
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Spam Probability",
                        f"{ensemble_prob:.2%}",
                        delta=f"{ensemble_prob - 0.5:.2%}" if ensemble_prob > 0.5 else f"{0.5 - ensemble_prob:.2%}"
                    )
                
                with col2:
                    classification = "SPAM" if ensemble_prob > 0.5 else "HAM"
                    color = "red" if classification == "SPAM" else "green"
                    st.markdown(f"**Classification:** <span style='color: {color}'>{classification}</span>", unsafe_allow_html=True)
                
                with col3:
                    confidence = max(ensemble_prob, 1 - ensemble_prob)
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Show individual model predictions
                st.write("**Individual Model Predictions:**")
                model_predictions = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Spam Probability': [predictions[model][0] for model in predictions.keys()]
                })
                st.dataframe(model_predictions, use_container_width=True)
                
                # Feature analysis
                st.write("**Feature Analysis:**")
                features = self.feature_extractor.extract_all_features([user_text])
                important_features = features.iloc[0].sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=important_features.values,
                    y=important_features.index,
                    orientation='h',
                    title="Top Features for This Text",
                    color=important_features.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to classify.")
    
    def create_statistical_summary(self, df: pd.DataFrame):
        """Create statistical summary of the dataset"""
        st.subheader("📋 Statistical Summary")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Statistics:**")
            stats = {
                'Total Messages': len(df),
                'Spam Messages': len(df[df['label'] == 'spam']),
                'Ham Messages': len(df[df['label'] == 'ham']),
                'Spam Percentage': f"{len(df[df['label'] == 'spam']) / len(df) * 100:.1f}%"
            }
            
            for key, value in stats.items():
                st.metric(key, value)
        
        with col2:
            st.write("**Text Statistics:**")
            text_stats = {
                'Average Length': f"{df['text'].str.len().mean():.1f} characters",
                'Max Length': f"{df['text'].str.len().max()} characters",
                'Min Length': f"{df['text'].str.len().min()} characters",
                'Average Words': f"{df['text'].str.split().str.len().mean():.1f} words"
            }
            
            for key, value in text_stats.items():
                st.metric(key, value)
        
        # Detailed statistics table
        st.write("**Detailed Statistics by Class:**")
        detailed_stats = df.groupby('label')['text'].agg([
            ('Count', 'count'),
            ('Avg Length', lambda x: x.str.len().mean()),
            ('Max Length', lambda x: x.str.len().max()),
            ('Min Length', lambda x: x.str.len().min()),
            ('Avg Words', lambda x: x.str.split().str.len().mean())
        ]).round(2)
        
        st.dataframe(detailed_stats, use_container_width=True)


def main():
    """Main function to run the Streamlit dashboard"""
    st.set_page_config(
        page_title="Email Spam Classifier Dashboard",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📧 Advanced Email Spam Classifier Dashboard")
    st.markdown("---")
    
    # Initialize dashboard
    dashboard = SpamVisualizationDashboard()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Dataset Overview", "Spam Analysis", "Feature Analysis", "Real-time Detection", "Statistical Summary"]
    )
    
    # Load data
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']].dropna()
        df.columns = ['label', 'text']
    except FileNotFoundError:
        st.error("Dataset file 'spam.csv' not found. Please ensure the file is in the current directory.")
        return
    
    # Load model if available
    try:
        classifier = AdvancedSpamClassifier()
        classifier.load_model('spam_classifier_model.pkl')
        model_loaded = True
    except FileNotFoundError:
        st.warning("Model not found. Please train the model first by running 'email_spam_classifier.py'")
        model_loaded = False
    
    # Display selected page
    if page == "Dataset Overview":
        dashboard.create_dataset_overview(df)
        dashboard.create_word_clouds(df)
    
    elif page == "Spam Analysis":
        dashboard.analyze_spam_patterns(df)
    
    elif page == "Feature Analysis":
        dashboard.create_feature_correlation_heatmap(df)
    
    elif page == "Real-time Detection":
        if model_loaded:
            dashboard.create_real_time_prediction_interface(classifier)
        else:
            st.error("Model not loaded. Please train the model first.")
    
    elif page == "Statistical Summary":
        dashboard.create_statistical_summary(df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Advanced Email Spam Classifier** | Built with Python, scikit-learn, and Streamlit"
    )


if __name__ == "__main__":
    main()
