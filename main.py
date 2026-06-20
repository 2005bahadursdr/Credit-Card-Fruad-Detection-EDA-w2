import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from src import eda

st.set_page_config(page_title="Credit Card Fraud Detection EDA", layout="wide")

st.title("💳 Credit Card Fraud Detection EDA Dashboard")
st.markdown("Explore anonymized Kaggle transaction data focusing on outlier detection and pattern recognition.")

# Sidebar navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Analysis View:",
    [
        "Data Overview",
        "Time-Series Analysis",
        "Correlation & Heatmaps",
        "Outlier & Anomaly Detection",
        "Distributions (Histograms & KDE)",
        "PCA & Clustering",
        "Demographic & Economic Stratification",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Upload dataset in 'Data Overview' to enable full analysis features.")

# Data loading and sampling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/raw/creditcard.csv")
        # Optimization: Sample majority class to prevent UI lag (5% of normal transactions)
        fraud = df[df['Class'] == 1]
        normal = df[df['Class'] == 0].sample(frac=0.05, random_state=42)
        sampled_df = pd.concat([fraud, normal]).sample(frac=1, random_state=42).reset_index(drop=True)
        return df, sampled_df
    except FileNotFoundError:
        st.error("Data file not found at `data/raw/creditcard.csv`.")
        return pd.DataFrame(), pd.DataFrame()

df, sampled_df = load_data()

if df.empty:
    st.stop()

if menu == "Data Overview":
    st.header("Data Overview")
    st.write("Initial view of the dataset properties, missing values, and data types.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Cases", len(df[df['Class'] == 1]))
    col3.metric("Normal Cases", len(df[df['Class'] == 0]))

    st.subheader("Dataset Sample (Original)")
    st.dataframe(df.head())

    st.subheader("Class Imbalance")
    fig = eda.plot_class_imbalance(df)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Time-Series Analysis":
    st.header("Time-Series Analysis")
    st.write("Analyzing fraud patterns over time (seconds elapsed since first transaction).")
    
    fig1 = eda.plot_transactions_over_time(df)
    st.plotly_chart(fig1, use_container_width=True)

elif menu == "Correlation & Heatmaps":
    st.header("Correlation & Heatmaps")
    st.write("Understanding relationships between V1-V28 features, Amount, and Class.")
    
    # We use the sampled dataframe for correlation to speed up calculation slightly
    # and provide clearer signals if the dataset is more balanced
    fig, corr = eda.plot_correlation_heatmap(sampled_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top Correlated Features with Class")
    fig2 = eda.plot_top_correlations(corr)
    st.plotly_chart(fig2, use_container_width=True)

elif menu == "Outlier & Anomaly Detection":
    st.header("Outlier & Anomaly Detection")
    st.write("Visualizing outliers using boxplots and statistical boundaries.")
    
    feature_to_plot = st.selectbox("Select Feature to Plot", [col for col in df.columns if col not in ['Class', 'Time']])
    
    # Using sampled_df to prevent browser from freezing with too many outliers drawn
    fig = eda.plot_outlier_boxplot(sampled_df, feature_to_plot)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Distributions (Histograms & KDE)":
    st.header("Distributions, KDE")
    st.write("Comparing the distribution of features between Fraud and Normal transactions.")
    
    feature_to_plot = st.selectbox("Select Feature", [col for col in df.columns if col not in ['Class']])
    
    fig = eda.plot_distributions(df, feature_to_plot)
    st.pyplot(fig)

elif menu == "PCA & Clustering":
    st.header("PCA & Clustering")
    st.write("Exploratory segmentation of transactions using Principal Component Analysis.")
    
    st.warning("Running PCA on sampled dataset (all fraud + 5% of normal) to optimize performance.")
    
    fig = eda.plot_pca_2d(sampled_df)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Demographic & Economic Stratification":
    st.header("Economic Stratification (Amount Based)")
    st.write("Since data is anonymized, we use the `Amount` variable as an economic proxy to analyze fraud likelihood across transaction sizes.")
    
    fig, fraud_rate_df = eda.get_amount_stratification(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate fraud percentage per strata
    st.subheader("Fraud Rate per Category")
    st.dataframe(fraud_rate_df)

st.markdown("---")
st.markdown("Developed for IT Security & Fraud Detection EDA")
