import streamlit as st
import pickle
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

# ========== Data & Model Loaders ==========

def load_data():
    return pd.read_csv("data/baseline_model.csv")

def load_xgb_model():
    with open('model/model_2.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# ========== Preprocessing ==========

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

    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[MODEL_FEATURES]

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

# ========== Streamlit UI ==========

st.set_page_config(page_title="Dashboard", layout="wide")
col1, col2 = st.columns([6,1])
with col1:
    st.title("📊 Dashboard for Churn Analysis")
    st.write("This dashboard shows how each feature affects customer churn.")
with col2:
    st.image("assets/logo.png",width=100)
st.divider()







data = load_data()
if 'subscription_status' not in data.columns and 'churn' in data.columns:
    data['subscription_status'] = data['churn'].map({0: 'No Churn', 1: 'Churn'})

feature = st.selectbox("🔎 Select Feature to Analyze", [col for col in data.columns if col not in ['churn', 'subscription_status']])
analysis_type = st.radio("📊 Select Analysis Type", ["Univariate", "Bivariate"], horizontal=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)
if pd.api.types.is_numeric_dtype(data[feature]):
    with st.container(border=True):
        col1.metric("Mean", f"{data[feature].mean():.2f}")
        col2.metric("Median", f"{data[feature].median():.2f}")
        col3.metric("Std Dev", f"{data[feature].std():.2f}")
else:
    with st.container(border=True):
        col1.metric("Unique Values", data[feature].nunique())
        top_cat = data[feature].value_counts().idxmax()
        col2.metric("Most Common", str(top_cat))
        col3.metric("Total Count", len(data))

st.markdown("---")

if analysis_type == "Univariate":
    st.subheader("📌 Univariate Distribution")
    with st.container(border=True):
        if pd.api.types.is_numeric_dtype(data[feature]):
            fig = px.histogram(data, x=feature, nbins=30, title=f"{feature} Distribution", marginal="violin")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.box(data, y=feature, title=f"{feature} Boxplot")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            counts = data[feature].value_counts().reset_index()
            counts.columns = [feature, "count"]
            fig = px.bar(counts, x=feature, y="count", title=f"{feature} Count")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("🔁 Relationship with Churn")
    with st.container(border=True):
        if pd.api.types.is_numeric_dtype(data[feature]):
            fig = px.histogram(data, x=feature, color='subscription_status', barmode='overlay', opacity=0.7, title=f"{feature} by Churn")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.box(data, x='subscription_status', y=feature, color='subscription_status', title=f"{feature} vs Churn Category")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig = px.histogram(data, x=feature, color='subscription_status', barmode='group', title=f"{feature} vs Churn")
            st.plotly_chart(fig, use_container_width=True)

            churn_table = pd.crosstab(data[feature], data['subscription_status'], normalize='index') * 100
            st.markdown("#### 📊 Churn Rate by Category")
            st.dataframe(churn_table.style.format("{:.1f}%").background_gradient(axis=1, cmap="RdYlGn_r"))

st.markdown("---")