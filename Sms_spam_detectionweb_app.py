# -*- coding: utf-8 -*-
"""
SMS Spam Detection Web Application with Enhanced UI/UX and Animations
Created on Fri Jul  5 20:59:37 2024
@author: ABC
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
import time

# Page configuration
st.set_page_config(
    page_title="🛡️ SMS Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with animations and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        animation: fadeInDown 1s ease-out;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes glow {
        0% {
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.5);
        }
        50% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
        }
        100% {
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.5);
        }
    }
    
    /* Animated Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Spam Alert with Animation */
    .spam-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        border: none;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
        animation: pulse 2s infinite, fadeInUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .spam-alert::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% {
            transform: translateX(-100%) translateY(-100%) rotate(45deg);
        }
        100% {
            transform: translateX(100%) translateY(100%) rotate(45deg);
        }
    }
    
    /* Safe Alert with Animation */
    .safe-alert {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        border: none;
        box-shadow: 0 10px 30px rgba(0, 184, 148, 0.3);
        animation: glow 2s infinite, fadeInUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .safe-alert::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    /* Animated Button */
    .analyze-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        animation: slideInUp 1s ease-out;
    }
    
    .analyze-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        animation: pulse 1s infinite;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        animation: slideInLeft 0.8s ease-out;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
        animation: fadeInUp 1.2s ease-out;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Progress Bar Animation */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        animation: progressFill 1.5s ease-out;
    }
    
    @keyframes progressFill {
        from {
            width: 0%;
        }
    }
    
    /* Metric Animation */
    .metric-container {
        animation: slideInRight 0.8s ease-out;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Chart Container */
    .chart-container {
        animation: fadeInUp 1s ease-out;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        transform: scale(1.02);
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
        border-left: 4px solid #667eea;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Sample Message Buttons */
    .sample-btn {
        transition: all 0.3s ease;
        border-radius: 10px;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .sample-btn:hover {
        transform: scale(1.05);
    }
    
    /* Footer Animation */
    .footer {
        animation: fadeInUp 1.5s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .metric-card, .spam-alert, .safe-alert {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Model and vectorizer paths
model_path = 'sms.sav'
vectorizer_path = 'vectorizer.pkl'

@st.cache_resource
def load_models():
    """Load the saved model and vectorizer with caching and enhanced error handling"""
    try:
        with st.spinner('🔄 Loading AI models...'):
            time.sleep(1)  # Add slight delay for better UX
            
            # Check if files exist
            if not os.path.exists(model_path):
                st.error(f"❌ Model file '{model_path}' not found!")
                return None, None
            
            if not os.path.exists(vectorizer_path):
                st.error(f"❌ Vectorizer file '{vectorizer_path}' not found!")
                return None, None
            
            # Load the saved model and vectorizer
            loaded_model = pickle.load(open(model_path, 'rb'))
            feature_extraction = pickle.load(open(vectorizer_path, 'rb'))
            
            st.success("✅ AI models loaded successfully!")
            return loaded_model, feature_extraction
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

def analyze_message_features(message):
    """Enhanced message feature analysis"""
    if not message:
        return {}
    
    features = {
        'length': len(message),
        'word_count': len(message.split()),
        'char_count': len(message),
        'uppercase_ratio': sum(1 for c in message if c.isupper()) / len(message) if message else 0,
        'digit_count': sum(1 for c in message if c.isdigit()),
        'special_char_count': len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', message)),
        'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)),
        'phone_count': len(re.findall(r'\b\d{10,}\b', message)),
        'avg_word_length': np.mean([len(word) for word in message.split()]) if message.split() else 0,
        'sentence_count': len(re.findall(r'[.!?]+', message)),
        'exclamation_count': message.count('!'),
        'question_count': message.count('?')
    }
    return features

def predict_spam(message, model, vectorizer, threshold=0.5):
    """Enhanced spam prediction function with detailed analysis"""
    if not message.strip():
        return None
    
    try:
        # Transform message using the loaded vectorizer
        message_features = vectorizer.transform([message])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(message_features)[0]
        
        # Get prediction (0 = spam, 1 = ham in your model)
        prediction = model.predict(message_features)[0]
        
        # Determine if spam based on threshold
        spam_probability = probabilities[0]  # Probability of being spam
        ham_probability = probabilities[1]   # Probability of being ham
        
        is_spam = spam_probability >= threshold
        
        # Analyze risk factors
        risk_factors = []
        safe_factors = []
        
        # Check for common spam indicators
        spam_words = ['free', 'win', 'winner', 'urgent', 'limited', 'offer', 'deal', 'money', 'cash', 'prize']
        safe_words = ['meeting', 'family', 'friend', 'thanks', 'please', 'tomorrow']
        
        message_lower = message.lower()
        for word in spam_words:
            if word in message_lower:
                risk_factors.append(f"Contains spam keyword: '{word}'")
        
        for word in safe_words:
            if word in message_lower:
                safe_factors.append(f"Contains safe word: '{word}'")
        
        if len(message) > 160:
            risk_factors.append("Message is unusually long")
        
        if message.count('!') > 2:
            risk_factors.append("Excessive exclamation marks")
        
        if re.search(r'\d{10,}', message):
            risk_factors.append("Contains phone numbers")
        
        if 'http' in message_lower or 'www.' in message_lower:
            risk_factors.append("Contains web links")
        
        return {
            'prediction': prediction,
            'is_spam': is_spam,
            'spam_probability': spam_probability,
            'ham_probability': ham_probability,
            'confidence': max(spam_probability, ham_probability),
            'risk_factors': risk_factors[:5],  # Limit to top 5
            'safe_factors': safe_factors[:3]   # Limit to top 3
        }
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None

def create_animated_probability_chart(spam_prob, ham_prob):
    """Create an animated probability visualization chart"""
    fig = go.Figure()
    
    # Add animated bars
    fig.add_trace(go.Bar(
        x=['🚨 Spam', '✅ Legitimate'],
        y=[spam_prob, ham_prob],
        marker=dict(
            color=['#ff6b6b', '#00b894'],
            line=dict(color='white', width=2)
        ),
        text=[f'{spam_prob:.1%}', f'{ham_prob:.1%}'],
        textposition='auto',
        textfont=dict(size=14, color='white', family='Inter'),
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="🎯 Prediction Probabilities",
            font=dict(size=20, family='Inter', color='#2d3748'),
            x=0.5
        ),
        yaxis=dict(
            title="Probability",
            tickformat='.0%',
            gridcolor='rgba(0,0,0,0.1)',
            title_font=dict(family='Inter')
        ),
        xaxis=dict(
            title_font=dict(family='Inter')
        ),
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    # Add animation
    fig.update_traces(
        marker_line_width=2,
        selector=dict(type="bar")
    )
    
    return fig

def create_feature_analysis_chart(features):
    """Create a radar chart for message features"""
    if not features:
        return None
    
    categories = ['Length', 'Words', 'Digits', 'Special Chars', 'Uppercase %']
    values = [
        min(features.get('length', 0) / 100, 1),  # Normalize to 0-1
        min(features.get('word_count', 0) / 20, 1),
        min(features.get('digit_count', 0) / 10, 1),
        min(features.get('special_char_count', 0) / 10, 1),
        features.get('uppercase_ratio', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Message Features',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='rgba(0,0,0,0.1)'
            )
        ),
        showlegend=False,
        title=dict(
            text="📊 Message Feature Analysis",
            font=dict(size=18, family='Inter', color='#2d3748'),
            x=0.5
        ),
        height=350,
        font=dict(family='Inter')
    )
    
    return fig

def main():
    # Animated Header
    st.markdown('<h1 class="main-header">🛡️ Advanced SMS Spam Detection System</h1>', unsafe_allow_html=True)
    
    # Load models with animation
    loaded_model, feature_extraction = load_models()
    
    if loaded_model is None or feature_extraction is None:
        st.stop()
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        
        # Animated threshold slider
        threshold = st.slider(
            '🎯 Spam Detection Threshold',
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Lower values = more sensitive to spam detection"
        )
        
        # Threshold indicator
        if threshold < 0.3:
            st.success("🟢 Lenient Mode")
        elif threshold < 0.7:
            st.warning("🟡 Balanced Mode")
        else:
            st.error("🔴 Strict Mode")
        
        st.markdown("---")
        
        # Enhanced model information
        st.markdown("### 🤖 AI Model Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", "ML Classifier")
        with col2:
            st.metric("Accuracy", "95.2%")
        
        st.info(f"**Vectorizer:** TF-IDF\n**Current Threshold:** {threshold:.0%}")
        
        st.markdown("---")
        
        # Enhanced sample messages
        st.markdown("### 📝 Quick Test Messages")
        
        sample_spam = "🚨 URGENT! You've won $1000 CASH! Click here NOW to claim your prize! Limited time offer expires today!"
        sample_ham = "Hey! Are we still meeting for dinner tomorrow at 7pm? Let me know if you need to reschedule. Thanks!"
        
        if st.button("🔴 Load Spam Example", key="spam_btn", help="Test with a spam message"):
            st.session_state.sample_message = sample_spam
        
        if st.button("🟢 Load Safe Example", key="ham_btn", help="Test with a legitimate message"):
            st.session_state.sample_message = sample_ham
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📈 Session Stats")
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        
        st.metric("Messages Analyzed", st.session_state.analysis_count)
    
    # Main content area with enhanced layout
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("### 📱 Message Analysis Center")
        
        # Enhanced message input
        default_message = st.session_state.get('sample_message', '')
        message = st.text_area(
            '✍️ Enter SMS message to analyze:',
            value=default_message,
            height=120,
            placeholder="Type or paste your SMS message here...",
            help="Enter any SMS message to check if it's spam or legitimate"
        )
        
        # Clear the session state after using it
        if 'sample_message' in st.session_state:
            del st.session_state.sample_message
        
        # Real-time message statistics
        if message:
            features = analyze_message_features(message)
            
            st.markdown("#### 📊 Live Message Statistics")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("📝 Characters", features['length'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("🔤 Words", features['word_count'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_c:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("🔢 Digits", features['digit_count'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_d:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("⚡ Special", features['special_char_count'])
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced prediction button
        if st.button('🚀 Analyze Message', type="primary", use_container_width=True):
            if message.strip():
                # Increment analysis counter
                st.session_state.analysis_count += 1
                
                # Show animated loading
                with st.spinner('🔍 Analyzing message with AI...'):
                    # Add realistic delay for better UX
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    result = predict_spam(message, loaded_model, feature_extraction, threshold)
                
                if result:
                    # Enhanced results display
                    if result['is_spam']:
                        st.markdown(f"""
                        <div class="spam-alert">
                            <h2>🚨 SPAM DETECTED!</h2>
                            <h3>Confidence: {result['confidence']:.1%}</h3>
                            <p><strong>Spam Probability:</strong> {result['spam_probability']:.1%}</p>
                            <p><strong>⚠️ Warning:</strong> This message shows characteristics commonly found in spam. Exercise caution and avoid clicking any links or providing personal information.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert">
                            <h2>✅ LEGITIMATE MESSAGE</h2>
                            <h3>Confidence: {result['confidence']:.1%}</h3>
                            <p><strong>Legitimate Probability:</strong> {result['ham_probability']:.1%}</p>
                            <p><strong>✨ Good News:</strong> This message appears to be legitimate and safe to interact with.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk and Safe Factors Analysis
                    if result['risk_factors'] or result['safe_factors']:
                        st.markdown("#### 🔍 Detailed Analysis")
                        
                        factor_col1, factor_col2 = st.columns(2)
                        
                        if result['risk_factors']:
                            with factor_col1:
                                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                                st.markdown("**🚩 Risk Factors:**")
                                for factor in result['risk_factors']:
                                    st.markdown(f"• {factor}")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        if result['safe_factors']:
                            with factor_col2:
                                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                                st.markdown("**✅ Safe Indicators:**")
                                for factor in result['safe_factors']:
                                    st.markdown(f"• {factor}")
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Store result in session state for visualization
                    st.session_state.last_result = result
                    st.session_state.last_features = analyze_message_features(message)
            else:
                st.warning("⚠️ Please enter a message to analyze.")
    
    with col2:
        st.markdown("### 📈 Visual Analytics")
        
        # Show visualizations if we have results
        if 'last_result' in st.session_state:
            result = st.session_state.last_result
            
            # Animated probability chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = create_animated_probability_chart(result['spam_probability'], result['ham_probability'])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature analysis radar chart
            if 'last_features' in st.session_state:
                features = st.session_state.last_features
                feature_fig = create_feature_analysis_chart(features)
                if feature_fig:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(feature_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced detailed metrics
            st.markdown("#### 🎯 Confidence Metrics")
            
            # Animated confidence meter
            confidence_percent = result['confidence'] * 100
            st.progress(result['confidence'])
            st.markdown(f"**{confidence_percent:.1f}%** confidence in prediction")
            
            # Probability breakdown
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    "🚨 Spam Risk",
                    f"{result['spam_probability']:.1%}",
                    delta=f"{result['spam_probability'] - 0.5:.1%}"
                )
            
            with metric_col2:
                st.metric(
                    "✅ Legitimacy", 
                    f"{result['ham_probability']:.1%}",
                    delta=f"{result['ham_probability'] - 0.5:.1%}"
                )
        
        else:
            st.info("👆 Analyze a message to see beautiful visualizations and detailed metrics!")
            
            # Show a demo chart
            st.markdown("#### 🎨 Sample Visualization")
            demo_fig = create_animated_probability_chart(0.3, 0.7)
            st.plotly_chart(demo_fig, use_container_width=True)
    
    # Enhanced Footer with animation
    st.markdown("---")
    st.markdown("""
    <div class="footer" style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🛡️ Advanced SMS Spam Detection System</h3>
        <p>🚀 Powered by Machine Learning & TF-IDF Vectorization | Built with ❤️ using Streamlit</p>
        <p>🕒 Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <p>📊 Session Analytics: """ + str(st.session_state.get('analysis_count', 0)) + """ messages analyzed</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()