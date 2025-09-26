import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    page_icon="📱",
    layout="wide"
)

# Title
st.title("📱 Telco Customer Churn Prediction Dashboard")
st.markdown("**Predict customer churn and analyze key risk factors**")

# Sidebar navigation
st.sidebar.header("🔍 Navigation")
page = st.sidebar.selectbox("Choose a page:", 
                           ["🎯 Churn Predictor", "📊 Data Analysis", "🤖 Model Performance"])

# Load data and model
@st.cache_data
def load_data():
    return pd.read_pickle('processed_telco_data.pkl')

@st.cache_resource
def load_model():
    return pickle.load(open('churn_model.pkl', 'rb'))

# Load everything
df = load_data()
model = load_model()

# Display basic info
st.sidebar.markdown("---")
st.sidebar.write(f"**Dataset:** {len(df):,} customers")
st.sidebar.write(f"**Churn Rate:** {df['Churn'].mean():.1%}")

# PAGE CONTENT
if page == "🎯 Churn Predictor":
    st.header("🎯 Individual Customer Churn Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Customer Information")
        
        # Simple, essential inputs only
        tenure = st.slider("Customer Tenure (months)", 1, 72, 24)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 119, 65)
        
        contract_type = st.selectbox("Contract Type", 
                                   ["Month-to-month", "One year", "Two year"])
        
        internet_service = st.selectbox("Internet Service", 
                                      ["DSL", "Fiber optic", "No Internet"])
        
        payment_method = st.selectbox("Payment Method", 
                                    ["Bank transfer (automatic)", 
                                     "Credit card (automatic)",
                                     "Electronic check", 
                                     "Mailed check"])
        
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    
    with col2:
        st.subheader("🔮 Prediction Results")
        
        # Calculate TotalCharges behind the scenes
        total_charges = monthly_charges * tenure
        
        # Create input in exact order your model expects
        input_data = [
            tenure,
            1 if internet_service == "Fiber optic" else 0,
            1 if internet_service == "No Internet" else 0,
            1 if contract_type == "One year" else 0,
            1 if contract_type == "Two year" else 0,
            1 if payment_method == "Electronic check" else 0,
            1 if payment_method == "Mailed check" else 0,
            1 if payment_method == "Credit card (automatic)" else 0,
            monthly_charges,
            total_charges,
            1 if paperless_billing == "Yes" else 0
        ]
        
        # Get prediction
        input_df = pd.DataFrame([input_data], columns=[
            'tenure', 'InternetService_Fiber optic', 'InternetService_No',
            'Contract