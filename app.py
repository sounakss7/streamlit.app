import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Streamlit UI
st.title("🛑 Fraud Detection using DBSCAN")
st.write("Upload a transaction dataset and detect anomalies using DBSCAN.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    
    # Select relevant columns (example: 'Amount' and 'Transaction Frequency')
    if 'Amount' in df.columns and 'Transaction Frequency' in df.columns:
        X = df[['Amount', 'Transaction Frequency']]
    else:
        st.error("Dataset must contain 'Amount' and 'Transaction Frequency' columns.")
        st.stop()
    
    # Standardize Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # DBSCAN Parameters
    eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.slider("Min Samples", 1, 20, 5)
    
    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(X_scaled)
    
    # Mark anomalies (Cluster -1)
    df['Fraud'] = df['Cluster'].apply(lambda x: "Fraud" if x == -1 else "Legit")
    
    st.write("### Fraud Detection Results", df.head(20))
    
    # Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Fraud', palette={'Legit': 'blue', 'Fraud': 'red'}, ax=ax)
    st.pyplot(fig)
    
    # Download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "fraud_results.csv", "text/csv")