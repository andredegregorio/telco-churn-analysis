import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# Load your actual data
@st.cache_data
def load_data():
    # Replace with your actual data loading path
    # df = pd.read_csv('your_telco_data.csv')
    # For now, you'll need to replace this with your actual data loading
    st.warning("Update the load_data() function with your actual data file path")
    return None

# Load and train your actual model
@st.cache_resource
def load_model():
    # This should match your actual model training process
    df = load_data()
    if df is None:
        return None
    
    # Your feature set - update to match your final model
    features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'PaperlessBilling',
        'InternetService_Fiber optic', 'InternetService_No',        
        'Contract_One year', 'Contract_Two year',                   
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 
        'PaymentMethod_Credit card (automatic)'
    ]
    
    X = df[features]
    y = df['Churn']
    
    # Train with your optimal settings
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=22,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced'
    )
    
    model.fit(X, y)
    return model, features

# Load data and model
df = load_data()
if df is not None:
    model, feature_names = load_model()
else:
    st.error("Please update the data loading function with your actual data path")
    st.stop()

# PAGE 1: CHURN PREDICTOR
if page == "🎯 Churn Predictor":
    st.header("🎯 Individual Customer Churn Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Customer Information")
        
        # Input features based on your actual model
        tenure = st.slider("Customer Tenure (months)", 1, 72, 24)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 119.0, 65.0)
        total_charges = st.slider("Total Charges ($)", 18.0, 8700.0, 2000.0)
        
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        
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
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
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
        prediction_proba = model.predict_proba(input_df)[0][1]
        
        # Use your optimal threshold (0.4)
        optimal_threshold = 0.4
        prediction = 1 if prediction_proba >= optimal_threshold else 0
        
        # Display prediction with your threshold
        if prediction == 1:
            st.error(f"⚠️ HIGH CHURN RISK")
            st.error(f"Churn Probability: {prediction_proba:.1%}")
        else:
            st.success(f"✅ LOW CHURN RISK")
            st.success(f"Churn Probability: {prediction_proba:.1%}")
        
        # Risk level based on your analysis
        if prediction_proba >= 0.7:
            risk_level = "🔴 CRITICAL"
        elif prediction_proba >= optimal_threshold:
            risk_level = "🟡 MODERATE"
        else:
            risk_level = "🟢 LOW"
        
        st.metric("Risk Level", risk_level)
        st.caption(f"Using {optimal_threshold:.0%} threshold for optimal business impact")
        
        # Recommendations based on your findings
        st.subheader("💡 Recommendations")
        if prediction_proba >= optimal_threshold:
            st.write("**Immediate Actions:**")
            if input_data['PaymentMethod_Electronic check'] == 1:
                st.write("• Offer automatic payment incentive (high-risk payment method)")
            if input_data['InternetService_Fiber optic'] == 1:
                st.write("• Review fiber service satisfaction (unexpectedly high churn)")
            if input_data['Contract_One year'] == 0 and input_data['Contract_Two year'] == 0:
                st.write("• Offer contract upgrade with benefits (month-to-month high risk)")
            if tenure < 12:
                st.write("• Implement new customer retention program (early churn risk)")
            if monthly_charges > 80:
                st.write("• Consider pricing review or value-add services")

# PAGE 2: DATA ANALYSIS  
elif page == "📊 Data Analysis":
    st.header("📊 Churn Analysis Dashboard")
    
    # Key metrics from your actual analysis
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Overall Churn Rate", f"{df['Churn'].mean():.1%}")
    with col3:
        st.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")
    with col4:
        # High risk calculation based on your findings
        high_risk = len(df[(df['PaymentMethod_Electronic check'] == 1) & 
                          (df['Contract_One year'] == 0) & (df['Contract_Two year'] == 0)])
        st.metric("High Risk Customers", f"{high_risk:,}")
    
    # Analysis sections based on your EDA
    st.subheader("🔍 Churn by Key Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Contract analysis from your results
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
        
        # Show actual numbers from your analysis
        st.caption("Month-to-month: ~43% churn | One year: ~11% churn | Two year: ~3% churn")
    
    with col2:
        # Internet service analysis from your results
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
        
        st.caption("Fiber optic: ~42% churn | DSL: ~19% churn | No Internet: ~7% churn")
    
    # Payment method analysis
    st.subheader("💳 Payment Method Impact")
    df['Payment_Method'] = 'Bank transfer (automatic)'
    df.loc[df['PaymentMethod_Electronic check'] == 1, 'Payment_Method'] = 'Electronic check'
    df.loc[df['PaymentMethod_Mailed check'] == 1, 'Payment_Method'] = 'Mailed check'
    df.loc[df['PaymentMethod_Credit card (automatic)'] == 1, 'Payment_Method'] = 'Credit card (automatic)'
    
    payment_churn = df.groupby('Payment_Method')['Churn'].mean().sort_values(ascending=False)
    
    fig_payment = px.bar(
        x=payment_churn.index, 
        y=payment_churn.values,
        title="Churn Rate by Payment Method",
        color=payment_churn.values,
        color_continuous_scale="RdYlGn_r"
    )
    fig_payment.update_layout(showlegend=False)
    st.plotly_chart(fig_payment, use_container_width=True)
    st.caption("Electronic check customers show significantly higher churn rates")

# PAGE 3: MODEL PERFORMANCE
elif page == "🤖 Model Performance":
    st.header("🤖 Model Performance Analysis")
    
    # Your actual model metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "76.9%")
    with col2:
        st.metric("AUC Score", "0.837")
    with col3:
        st.metric("Churn Recall", "73%")
    
    # Threshold optimization results
    st.subheader("🎯 Threshold Optimization")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Optimal Threshold", "0.4", help="Balances recall and operational feasibility")
        st.metric("Contact Rate", "41%", help="Percentage of customers to contact")
    with col2:
        st.metric("Churn Detection", "81%", help="Percentage of churners caught")
        st.metric("Precision", "52%", help="Accuracy of churn predictions")
    
    # Feature importance
    st.subheader("📊 Feature Importance")
    
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
    
    # Model insights based on your analysis
    st.subheader("🔍 Key Model Insights")
    st.write("**Top Predictive Factors:**")
    st.write("1. **Tenure** - Most important predictor (longer tenure = lower churn)")
    st.write("2. **Contract Type** - Month-to-month contracts show 43% churn vs 3% for two-year") 
    st.write("3. **Payment Method** - Electronic check users show 45% churn rate")
    st.write("4. **Internet Service** - Fiber optic users unexpectedly show higher churn")
    
    st.write("**Business Impact:**")
    st.write("• Model catches 81% of churners while contacting 41% of customers")
    st.write("• Improved recall by 21 percentage points over baseline model")
    st.write("• Optimal threshold provides strong ROI for retention campaigns")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit | Telco Churn Prediction | Random Forest Model*")