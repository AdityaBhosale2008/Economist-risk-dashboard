import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import os

from dotenv import load_dotenv
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


llm = ChatOpenAI(
    openai_api_base="https://integrate.api.nvidia.com/v1",
    openai_api_key=NVIDIA_API_KEY,
    model="nvidia/llama-3.1-nemotron-ultra-253b-v1"
)

def prompt(feature_shap_importance: dict ,proba: int): 
    system_message = f"""
                YOU ARE AN MACHINE LEARNING MODEL EXPLAINABILITY EXPERT
                Here are the details for a  is assisting:
                    - Dictioary of feature_name, shap_impact and feature_importance for xgboost machine learning model: {feature_shap_importance}
                    - Models Predicted Churn Probability: {proba}
                Based on this information, explain to the agent in non-technical terms:
                    1. Provide summary of who the customer is froe user context features and his current status of action context features.
                    2. What is the customers predicted churn status with probability? Use the following scale:
                        - Under 0.1 : Less likely
                        - 0.1 to 0.5 : Likely
                        - 0.5 to 0.7 : Very likely
                    - Above 0.7 : Highly likely
                    3. Identify the top 3 reasons for the customers potential churn. Provide a brief explanation of why these
                    features significantly influence the churn prediction.
                    4. Suggest the top 3 actions the agent can take to reduce the likelihood of churn, based on the feature impacts. Each suggestion should include:
                        - An explanation of why this action is expected to impact churn, based solely on the data provided.
                Remember :
                    - The magnitude of a SHAP value indicates the strength of a feature's influence on the prediction.
                    - Positive SHAP values increase the likelihood of churn; negative values decrease it.
                    - Feature Importances values adds up to 1, greater the value higher the feature is important in prediction.
                    - Recommendations should be strictly based on the information provided in the SHAP contributions and customer features.
            """
    return system_message

def escape_curly_braces(s):
    return s.replace("{", "{{").replace("}", "}}")

parser = StrOutputParser()





@st.cache_data
def load_data():
    return pd.read_csv("data/home_data.csv")

df = load_data()


# --- ML Model Functions (from playground) ---
def load_xgb_model():
    with open('model/model_2.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess(user_input):
    df = pd.DataFrame([user_input])
    df['subscription_type'] = df['subscription_type'].map({'Espresso': 0, 'Digital': 1, 'Digital+Print': 2})
    df['plan_type'] = df['plan_type'].map({'Monthly': 0, 'Annual': 1})
    df['auto_renew'] = df['auto_renew'].map({'Yes': 1, 'No': 0})
    df['discount_used_last_renewal'] = df['discount_used_last_renewal'].map({'Yes': 1, 'No': 0})
    df['downgrade_history'] = df['downgrade_history'].map({'Yes': 1, 'No': 0})
    df['previous_renewal_status'] = df['previous_renewal_status'].map({'Auto': 1, 'Manual': 0})
    df['signup_source'] = df['signup_source'].map({'Web': 0, 'Mobile App': 0, 'Referral': 1})
    df = pd.get_dummies(df, columns=['region', 'most_read_category', 'primary_device', 'payment_method', 'last_campaign_engaged'])
    df = df.fillna(0)

    MODEL_FEATURES = ['subscription_type', 'plan_type', 'auto_renew',
        'avg_articles_per_week', 'days_since_last_login',
        'support_tickets_last_90d', 'discount_used_last_renewal',
        'email_open_rate', 'time_spent_per_session_mins',
        'completion_rate', 'article_skips_per_week',
        'previous_renewal_status', 'campaign_ctr', 'nps_score',
        'sentiment_score', 'csat_score', 'customer_age', 'signup_source',
        'downgrade_history', 'tenure_days', 'region_Asia', 'region_Europe',
        'region_North America', 'region_Others', 'most_read_Culture',
        'most_read_Environment', 'most_read_Finance', 'most_read_Politics',
        'most_read_Technology', 'primary_device_Desktop',
        'primary_device_Mobile', 'primary_device_Tablet',
        'payment_method_Credit Card', 'payment_method_Debit Card',
        'payment_method_PayPal', 'last_campaign_engaged_Newsletter Promo',
        'last_campaign_engaged_Retention Offer',
        'last_campaign_engaged_Survey']


    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[MODEL_FEATURES]





# Check if the session state has the selected customer id

if 'selected_customer_id' not in st.session_state:
    st.write("No customer selected.")
else:
    customer_id = st.session_state['selected_customer_id']
    customer_row = df[df['customer_id'] == customer_id]
    if customer_row.empty:
        st.write("Customer not found.")
    else:
        customer = customer_row.iloc[0]
        col1, col2 = st.columns([6,1])
        with col1:
            st.title(f"🪪 Customer Profile for {customer['customer_id']}")
        with col2:
            st.image("assets/logo.png",width=100)        
        st.divider()

     # --- ML Prediction ---
        st.markdown("## 🤖 Churn Prediction for Customer")

        # Prepare input dict for preprocess
        input_dict = customer.to_dict()
        # Ensure correct string values for categorical fields
        input_dict['auto_renew'] = str(input_dict['auto_renew'])
        input_dict['discount_used_last_renewal'] = str(input_dict['discount_used_last_renewal'])
        input_dict['downgrade_history'] = str(input_dict['downgrade_history'])
        input_dict['previous_renewal_status'] = str(input_dict['previous_renewal_status'])
        input_dict['subscription_type'] = str(input_dict['subscription_type'])
        input_dict['plan_type'] = str(input_dict['plan_type'])
        input_dict['signup_source'] = str(input_dict['signup_source'])
        input_dict['region'] = str(input_dict['region'])
        input_dict['most_read_category'] = str(input_dict['most_read_category'])
        input_dict['primary_device'] = str(input_dict['primary_device'])
        input_dict['payment_method'] = str(input_dict['payment_method'])
        input_dict['last_campaign_engaged'] = str(input_dict['last_campaign_engaged'])

        # Preprocess and predict
        X = preprocess(input_dict)
        model = load_xgb_model()
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]



        feature_importances = model.feature_importances_
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        row = shap_values[0]
        shap_impact = row.values
        features = row.feature_names 

        feature_shap_importance = {
            feature: {
                'shap_impact': float(shap),
                'feature_importance': float(importance)
            }
            for feature, shap, importance in zip(features, shap_impact, feature_importances)
        }

        escaped_feature_shap_importance = escape_curly_braces(str(feature_shap_importance))
        prompt_template = prompt(escaped_feature_shap_importance, proba)
        prompt = ChatPromptTemplate.from_messages([
                    ("system", prompt_template),
                    ("human", "")
                ])
        


        col1, col2 = st.columns(2)

        with col1:
            if input_dict['churn_risk'] == "High":
                result = "High " + " " + "Risk of Churn"
                st.error(f"📊 **Prediction:** {result}")
            elif input_dict['churn_risk'] == "Medium":
                result = "Medium " + " " + "Risk of Churn"
                st.warning(f"📊 **Prediction:** {result}")
            elif input_dict['churn_risk'] == "Low":
                result = "Low " + " " + "Risk of Churn"
                st.success(f"📊 **Prediction:** {result}")
            else:
                st.error("📊 **Prediction:** Already Churned")    
            
            st.info(f"🧠 Model Confidence: **{proba * 100:.2f}%** for Churn")

            report_button = st.button("Get Report")       
        with col2: 
            # SHAP GRAPH
            with st.container(border=True):
                # SHAP Feature Importance
                st.subheader("🔎 Feature Importance")
                top_idx = np.argsort(np.abs(shap_impact))[-10:]
                top_features = [features[i] for i in top_idx]
                top_shap = shap_impact[top_idx]
                fig_waterfall = go.Figure(go.Waterfall(
                    orientation="h",
                    measure=["relative"] * len(top_features),
                    x=top_shap,
                    y=top_features,
                    text=[f"{v:.3f}" for v in top_shap],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    decreasing={"marker": {"color": "green"}},
                    increasing={"marker": {"color": "red"}},
                ))
                fig_waterfall.update_layout(
                    title="",
                    xaxis_title="SHAP Value Impact",
                    yaxis_title="Feature",
                    waterfallgap=0.4
                )
                st.plotly_chart(fig_waterfall, use_container_width=True)

            


        if report_button:
            with st.spinner("Generating Report..."):
                with st.container(border=True):
                    chain = prompt | llm | parser

                    result = chain.invoke({"feature_shap_importance": escaped_feature_shap_importance, "proba": proba})
                    st.write(f"### SHAP Analysis \n {result}")


        st.divider()


        # Personal Info
        with st.container(border=True):
            col_1, col_2, col_3 = st.columns([1.5,1.5,1])
            with col_1:
                st.subheader("👤 Personal Information")
                st.markdown(f"**🧑 Name:** {customer['first_name']} {customer['last_name']}")
                st.markdown(f"**🎂 Age:** {customer['customer_age']}")
                st.markdown(f"**🚻 Gender:** {customer['gender']}")
            with col_2:
                if customer['subscription_status'] == "Active":
                    st.success("✅ Subscription is Active")
                else:
                    st.error("❌ Subscription is Cancelled")
                st.markdown(f"**✉️ Email:** {customer['email']}")
                st.markdown(f"**📞 Phone:** {customer['Phone']}")
                st.markdown(f"**🌍 Region:** {customer['region']}")
            with col_3:
                if customer['gender'] == "Male" or customer['gender'] == "Other":
                    st.image("assets/man.jpg",width=300)
                else:
                    st.image("assets/woman.jpg",width=300)


        colm1, colm2 = st.columns(2)
        with colm1:
            # Subscription Info
            if customer['subscription_status'] == "Active":
                with st.container(border=True):
                    st.subheader("💳 Subscription Details")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**📦 Type:** {customer['subscription_type']}")
                        st.markdown(f"**🗓️ Plan:** {customer['plan_type']}")
                        if customer['auto_renew'] == "No":
                            auto_renewal = "Enabled"
                        else:
                            auto_renewal = "Disabled"
                        st.markdown(f"**🔁 Auto Renew:** {auto_renewal}")
                        st.markdown(f"**🟢 Start Date:** {customer['subscription_start_date']}")
                        st.markdown(f"**🏷️ Discount Used for Renewal:** {customer['discount_used_last_renewal']}")

                    with col2:
                        st.markdown(f"**💳 Payment Method:** {customer['payment_method']}")
                        st.markdown(f"**🛒 Signup Source:** {customer['signup_source']}")

                        st.markdown(f"**🔄 Previous Renewal Status:** {customer['previous_renewal_status']}")
                        st.markdown(f"**⬇️ Downgrade History:** {customer['downgrade_history']}")
            else:
                with st.expander(label="💳 Previous Subscription Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**📦 Type:** {customer['subscription_type']}")
                        st.markdown(f"**🗓️ Plan:** {customer['plan_type']}")
                        if customer['auto_renew'] == "No":
                            auto_renewal = "Enabled"
                        else:
                            auto_renewal = "Disabled"
                        st.markdown(f"**🔁 Auto Renew:** {auto_renewal}")
                        st.markdown(f"**🏷️ Discount Used Last Renewal:** {customer['discount_used_last_renewal']}")
                        st.markdown(f"**⬇️ Downgrade History:** {customer['downgrade_history']}")  
                    with col2:
                        st.markdown(f"**🟢 Start Date:** {customer['subscription_start_date']}")
                        st.markdown(f"**🔚 End Date:** {customer['subscription_end_date']}")
                        st.markdown(f"**💳 Payment Method:** {customer['payment_method']}")
                        st.markdown(f"**🛒 Signup Source:** {customer['signup_source']}")
                        st.markdown(f"**🔄 Previous Renewal Status:** {customer['previous_renewal_status']}")
        with colm2: 
            # User Activity
            with st.container(border=True):
                st.subheader("📱 User Activity")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**📚 Most Read Category:** {customer['most_read_category']}")
                    st.metric("📰 Avg Articles/Week", customer['avg_articles_per_week'])
                    st.metric("📆 Days Since Last Login", customer['days_since_last_login'])
                with col2:
                    st.markdown(f"**📱 Primary Device:** {customer['primary_device']}")
                    st.metric("📉 Article Skips/Week", customer['article_skips_per_week'])
                    st.metric("⏳ Time/Session (min)", customer['time_spent_per_session_mins'])

        colm3, colm4 = st.columns(2)
        with colm3: 
            #  Campaign Engagement Metrics
            with st.container(border=True):
                st.subheader("📊 Campaign Engagement Metrics")
                st.markdown(f"**📢 Last Campaign Engaged:** {customer['last_campaign_engaged']}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📬 Email Open Rate", f"{customer['email_open_rate']*100:.1f}%")
                with col2:
                    st.metric("📉 Campaign CTR", f"{customer['campaign_ctr']}")
                with col3:
                    st.metric("📈 Completion Rate", f"{customer['completion_rate']*100:.1f}%")
        with colm4:
            # Customer Support
            with st.container(border=True):
                st.subheader("🧰 Customer Support")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎟️ Support Tickets (90d)", customer['support_tickets_last_90d'])
                    st.metric("😊 CSAT Score", customer['csat_score'])
                with col2:
                    st.metric("💬 Sentiment", customer['sentiment_score'])
                    st.metric("🟢 NPS Score", customer['nps_score'])

        colm5, colm6 = st.columns(2)

        with colm5:
            with st.container(border=True):
                st.write("Sample Chart")
                st.markdown("#### Content Engagement Chart")
                # Sample content engagement chart (e.g., articles read per week)
                weeks = [f"Week {i+1}" for i in range(8)]
                articles = np.random.randint(1, 10, size=8)
                fig_content = px.bar(x=weeks, y=articles, labels={'x': 'Week', 'y': 'Articles Read'})
                st.plotly_chart(fig_content, use_container_width=True)
        
        with colm6:
            with st.container(border=True):
                st.write("Sample Chart")
                st.markdown("#### Campaign Engagement Chart")
                # Sample campaign engagement chart (e.g., email open rate over time)
                days = [f"Day {i+1}" for i in range(10)]
                open_rates = np.random.uniform(0.2, 1.0, size=10)
                fig_campaign = px.line(x=days, y=open_rates, labels={'x': 'Day', 'y': 'Email Open Rate'})
                st.plotly_chart(fig_campaign, use_container_width=True)

        st.divider()
   
