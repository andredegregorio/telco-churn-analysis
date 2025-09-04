import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    page_icon="📱",
    layout="wide"
)

# Title and description
st.title("📱 Telco Customer Churn Prediction Dashboard")
st.markdown("**Predict customer churn and analyze key risk factors**")

# Sidebar for navigation
st.sidebar.header("🔍 Navigation")
page = st.sidebar.selectbox("Choose a page:", 
                           ["🎯 Churn Predictor", "📊 Data Analysis", "🤖 Model Performance"])

# Load data function (you'll need to adapt this to your data loading)
@st.cache_data
def load_data():
    # Replace this with your actual data loading
    # For demo, creating sample data structure
    np.random.seed(42)
    n_samples = 1000
    data = {
        'tenure': np.random.randint(1, 73, n_samples),
        'InternetService_Fiber optic': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'InternetService_No': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Contract_One year': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Contract_Two year': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'PaymentMethod_Electronic check': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'PaymentMethod_Mailed check': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'PaymentMethod_Credit card (automatic)': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    return pd.DataFrame(data)

# Load model function
@st.cache_resource
def load_model():
    # In real app, you'd load your trained model
    # For demo, training a simple model
    df = load_data()
    features = ['tenure', 'InternetService_Fiber optic', 'InternetService_No', 
               'Contract_One year', 'Contract_Two year', 'PaymentMethod_Electronic check', 
               'PaymentMethod_Mailed check', 'PaymentMethod_Credit card (automatic)']
    X = df[features]
    y = df['Churn']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Load data and model
df = load_data()
model = load_model()

# PAGE 1: CHURN PREDICTOR
if page == "🎯 Churn Predictor":
    st.header("🎯 Individual Customer Churn Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Customer Information")
        
        # Input features
        tenure = st.slider("Customer Tenure (months)", 1, 72, 24)
        
        internet_service = st.selectbox("Internet Service", 
                                      ["DSL", "Fiber optic", "No Internet"])
        
        contract_type = st.selectbox("Contract Type", 
                                   ["Month-to-month", "One year", "Two year"])
        
        payment_method = st.selectbox("Payment Method", 
                                    ["Bank transfer (automatic)", 
                                     "Credit card (automatic)",
                                     "Electronic check", 
                                     "Mailed check"])
    
    with col2:
        st.subheader("Prediction Results")
        
        # Convert inputs to model format
        input_data = {
            'tenure': tenure,
            'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
            'InternetService_No': 1 if internet_service == "No Internet" else 0,
            'Contract_One year': 1 if contract_type == "One year" else 0,
            'Contract_Two year': 1 if contract_type == "Two year" else 0,
            'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        }
        
        # Make prediction
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]
        
        # Display prediction
        if prediction == 1:
            st.error(f"⚠️ HIGH CHURN RISK")
            st.error(f"Churn Probability: {prediction_proba:.1%}")
        else:
            st.success(f"✅ LOW CHURN RISK")
            st.success(f"Churn Probability: {prediction_proba:.1%}")
        
        # Risk level
        if prediction_proba >= 0.7:
            risk_level = "🔴 CRITICAL"
        elif prediction_proba >= 0.5:
            risk_level = "🟡 MODERATE"
        else:
            risk_level = "🟢 LOW"
        
        st.metric("Risk Level", risk_level)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if prediction_proba >= 0.5:
            st.write("**Immediate Actions:**")
            if input_data['PaymentMethod_Electronic check'] == 1:
                st.write("• Offer automatic payment incentive")
            if input_data['InternetService_Fiber optic'] == 1:
                st.write("• Review fiber service satisfaction")
            if input_data['Contract_One year'] == 0 and input_data['Contract_Two year'] == 0:
                st.write("• Offer contract upgrade with benefits")
            if tenure < 12:
                st.write("• Implement new customer retention program")

# PAGE 2: DATA ANALYSIS
elif page == "📊 Data Analysis":
    st.header("📊 Churn Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Overall Churn Rate", f"{df['Churn'].mean():.1%}")
    with col3:
        st.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")
    with col4:
        high_risk = len(df[(df['PaymentMethod_Electronic check'] == 1) & 
                          (df['Contract_One year'] == 0) & (df['Contract_Two year'] == 0)])
        st.metric("High Risk Customers", f"{high_risk:,}")
    
    # Analysis sections
    st.subheader("🔍 Churn by Key Factors")
    
    # Contract analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Create contract type column for analysis
        df['Contract_Type'] = 'Month-to-month'
        df.loc[df['Contract_One year'] == 1, 'Contract_Type'] = 'One year'
        df.loc[df['Contract_Two year'] == 1, 'Contract_Type'] = 'Two year'
        
        contract_churn = df.groupby('Contract_Type')['Churn'].mean().sort_values(ascending=False)
        
        fig_contract = px.bar(
            x=contract_churn.index, 
            y=contract_churn.values,
            title="Churn Rate by Contract Type",
            color=contract_churn.values,
            color_continuous_scale="RdYlGn_r"
        )
        fig_contract.update_layout(showlegend=False)
        st.plotly_chart(fig_contract, use_container_width=True)
    
    with col2:
        # Internet service analysis
        df['Internet_Type'] = 'DSL'
        df.loc[df['InternetService_Fiber optic'] == 1, 'Internet_Type'] = 'Fiber optic'
        df.loc[df['InternetService_No'] == 1, 'Internet_Type'] = 'No Internet'
        
        internet_churn = df.groupby('Internet_Type')['Churn'].mean().sort_values(ascending=False)
        
        fig_internet = px.bar(
            x=internet_churn.index, 
            y=internet_churn.values,
            title="Churn Rate by Internet Service",
            color=internet_churn.values,
            color_continuous_scale="RdYlGn_r"
        )
        fig_internet.update_layout(showlegend=False)
        st.plotly_chart(fig_internet, use_container_width=True)
    
    # Tenure analysis
    st.subheader("📈 Churn Rate by Tenure")
    
    # Create tenure groups for better visualization
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 72], 
                               labels=['0-12m', '12-24m', '24-36m', '36-48m', '48-72m'])
    tenure_churn = df.groupby('Tenure_Group')['Churn'].mean()
    
    fig_tenure = px.line(
        x=tenure_churn.index, 
        y=tenure_churn.values,
        title="Churn Rate Decreases with Tenure",
        markers=True
    )
    fig_tenure.update_traces(line_color='red', line_width=3, marker_size=8)
    st.plotly_chart(fig_tenure, use_container_width=True)
    
    # Risk segments
    st.subheader("🚨 High-Risk Customer Segments")
    
    high_risk_segments = df[
        (df['PaymentMethod_Electronic check'] == 1) & 
        (df['Contract_One year'] == 0) & 
        (df['Contract_Two year'] == 0)
    ]['Churn'].mean()
    
    low_risk_segments = df[
        (df['PaymentMethod_Electronic check'] == 0) & 
        (df['Contract_Two year'] == 1)
    ]['Churn'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("High-Risk Segment", f"{high_risk_segments:.1%}", 
                 help="Month-to-month + Electronic check")
    with col2:
        st.metric("Low-Risk Segment", f"{low_risk_segments:.1%}", 
                 help="Two-year contract + Non-electronic payment")

# PAGE 3: MODEL PERFORMANCE
elif page == "🤖 Model Performance":
    st.header("🤖 Model Performance Analysis")
    
    # Model metrics (simulated - replace with actual metrics)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "77.8%")
    with col2:
        st.metric("AUC Score", "0.823")
    with col3:
        st.metric("Precision (Churn)", "60%")
    
    # Feature importance
    st.subheader("📊 Feature Importance")
    
    # Get feature importance from model
    feature_names = ['tenure', 'InternetService_Fiber optic', 'InternetService_No', 
                    'Contract_One year', 'Contract_Two year', 'PaymentMethod_Electronic check', 
                    'PaymentMethod_Mailed check', 'PaymentMethod_Credit card (automatic)']
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig_importance = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title="Feature Importance in Churn Prediction"
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model insights
    st.subheader("🔍 Key Model Insights")
    st.write("**Top Predictive Factors:**")
    st.write("1. **Tenure** - Most important predictor")
    st.write("2. **Contract Type** - Strong impact on retention") 
    st.write("3. **Payment Method** - Electronic check increases risk")
    st.write("4. **Internet Service** - Fiber optic users have higher churn")
    
    st.write("**Business Recommendations:**")
    st.write("• Focus retention efforts on new customers (< 12 months)")
    st.write("• Incentivize contract upgrades from month-to-month")
    st.write("• Promote automatic payment methods")
    st.write("• Investigate fiber optic service quality issues")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit | Data Science Project | Customer Churn Prediction*")