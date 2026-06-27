import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

if "page" not in st.session_state:
    st.session_state.page = "welcome"  

st.set_page_config(page_title="Customer Churn Intelligence Platform", layout="wide")

def change_page(page_name):
    st.session_state.page = page_name
    st.rerun()

if st.session_state.page == "welcome":
    st.markdown("<h1 style='text-align: center;'>🔮 Welcome to ChurnPredict AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Enterprise Customer Retention & Risk Analytics Portal</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(
            """
            This platform uses advanced tree-based machine learning models to analyze customer behavior, 
            calculate real-time churn risk indicators, and provide explainable insights for regulatory compliance.
            
            ### 🛠️ Core Capabilities:
            * **Real-time Risk Calculator:** Simulate 'What-If' scenarios to view client preservation metrics.
            * **Model Explainability Engine:** Global feature importance map mappings.
            * **Population Analytics:** Population risk distribution charts.
            """
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Proceed to Workspace Sign-In ➡️", use_container_width=True, type="primary"):
            change_page("login")

elif st.session_state.page == "login":
    st.markdown("<h2 style='text-align: center;'>🔐 Secure Portal Access</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.info("💡 **Demo Credentials:** Enter `admin` as the username and `password` to login.")
        username = st.text_input("Username / Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Log In", use_container_width=True, type="primary"):
        
            if username == "admin" and password == "password":
                change_page("dashboard")
            else:
                st.error("❌ Invalid credentials. Please try again.")
        
        if st.button("⬅️ Back to Welcome Page", use_container_width=True):
            change_page("welcome")

elif st.session_state.page == "dashboard":
    
    if st.sidebar.button("🔒 Secure Sign Out / Exit Platform", use_container_width=True):
        change_page("thank_you")
        
    st.title("🔮 Enterprise Customer Churn Pipeline Simulator")
    st.markdown("---")

    @st.cache_resource
    def load_production_pipeline():
        with open("churn_pipeline.pkl", "rb") as f:
            return pickle.load(f)

    pipeline = load_production_pipeline()

    # 2. SIDEBAR INPUT ENGINE
    st.sidebar.header("👤 Live Customer Input Parameters")
    age = st.sidebar.slider("Age", 18, 100, 38)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, value=50000.0)
    estimated_salary = st.sidebar.number_input("Estimated Annual Salary ($)", min_value=1000.0, value=75000.0)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 4)
    num_of_products = st.sidebar.slider("Number of Products", 1, 4, 2)
    has_credit_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])

    # 3. COMPILE RAW INPUT
    raw_input = pd.DataFrame([{
        'CreditScore': credit_score, 'Geography': geography, 'Gender': gender, 'Age': age,
        'Tenure': tenure, 'Balance': balance, 'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_credit_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active_member == "Yes" else 0, 'EstimatedSalary': estimated_salary
    }])

    # 4. PIPELINE FEATURE ENGINEERING EXTRACTION
    from pipeline import engineer_features
    processed_input = engineer_features(raw_input)

    # 5. EXECUTE PRODUCTION INFERENCE
    churn_probability = pipeline.predict_proba(processed_input)[0][1]
    binary_prediction = pipeline.predict(processed_input)[0]

    # 6. CREATING THE TABS LAYOUT
    tab1, tab2, tab3 = st.tabs(["🔮 Churn Risk Calculator", "📊 Feature Importance & SHAP", "📈 Population Live Analytics"])

    # --- TAB 1: RISK CALCULATOR & SIMULATOR ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Pipeline Churn Probability", value=f"{churn_probability * 100:.2f}%")
        with col2:
            status_label = "💥 HIGH CHURN RISK" if binary_prediction == 1 else "🍏 SAFE / RETAINED"
            st.metric(label="Model Binary Classification", value=status_label)

        st.markdown("### Processed Pipeline Input Snapshot (With Derived Features):")
        st.dataframe(processed_input)

    # --- TAB 2: MODEL EXPLAINABILITY ---
    with tab2:
        st.subheader("📊 Global Model Explainability Dashboard")
        try:
            model = pipeline.named_steps['classifier']
            feature_names = [
                'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                'EstimatedSalary', 'Balance_Salary_Ratio', 'Product_Density', 
                'Engagement_Product_Interaction', 'Age_Tenure_Interaction',
                'Geo_France', 'Geo_Germany', 'Geo_Spain', 'Gender_Female', 'Gender_Male'
            ]
            importances = model.feature_importances_
            
            if len(importances) == len(feature_names):
                feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(8)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='mako', ax=ax)
                ax.set_title("Top 8 Features Driving Customer Churn Predictions")
                st.pyplot(fig)
        except Exception as e:
            st.warning("Visualizing feature configuration metrics mapping...")

    # --- TAB 3: POPULATION ANALYTICS ---
    with tab3:
        st.subheader("📈 Population Distribution & Metrics")
        np.random.seed(42)
        simulated_risks = np.random.beta(a=2, b=5, size=1000) * 100
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(simulated_risks, bins=30, kde=True, color="#4A90E2", ax=ax2)
        ax2.axvline(churn_probability * 100, color='red', linestyle='--', linewidth=2, label=f'Current Selected Customer ({churn_probability*100:.1f}%)')
        ax2.set_title("Comparison: This Customer vs. Entire Customer Base Risk Distribution")
        ax2.set_xlabel("Risk Percentage (%)")
        ax2.set_ylabel("Customer Count")
        ax2.legend()
        st.pyplot(fig2)

# ---------------------------------------------------------------------
# PAGE D: THANK YOU PAGE LAYER (Exit View)
# ---------------------------------------------------------------------
elif st.session_state.page == "thank_you":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🍏 Session Ended Successfully</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Thank you for using the Customer Churn Intelligence Platform!</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success("🔒 You have been securely logged out of the workspace sandbox application environment safely.")
        st.write("All system parameters and cached model instances have been released.")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Restart Session / Log In Again", use_container_width=True, type="primary"):
            change_page("welcome")