import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore.csv")
    return df

df = load_data()

# Title
st.title("📱 Discovering the DNA of Viral Apps")

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Convert categorical and numerical values correctly
df['Installs'] = df['Installs'].str.replace(r'[\+,]', '', regex=True).astype(float)
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Size'] = df['Size'].str.replace(r'[^0-9]', '', regex=True)
df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
df['Price'] = df['Price'].str.replace(r'[\$,]', '', regex=True).astype(float)

# Fill NaN values with 0
df.fillna(0, inplace=True)

# Feature Engineering: Engagement Score
df['Engagement Score'] = (np.log1p(df['Reviews']) / np.log1p(df['Installs'] + 1)) * df['Rating']

# Standardizing numerical features for clustering
features = ['Engagement Score', 'Rating', 'Reviews', 'Installs']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Apply K-Means Clustering
optimal_k = 3  # Chosen based on Elbow Method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Mapping Cluster Labels
cluster_labels = {0: 'Highly Viral 🚀', 1: 'Moderately Popular 📈', 2: 'Low Engagement 📉'}

df['Cluster Label'] = df['Cluster'].map(cluster_labels)

# Streamlit User Input for Prediction
st.sidebar.header("🔍 Predict Your App's Virality")

installs = st.sidebar.number_input("📥 Total Installs", min_value=0, value=10000)
reviews = st.sidebar.number_input("📝 Total Reviews", min_value=0, value=100)
rating = st.sidebar.slider("⭐ App Rating", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
size = st.sidebar.number_input("📏 App Size (MB)", min_value=0, value=50)
price = st.sidebar.number_input("💲 Price (if paid)", min_value=0.0, value=0.0)

if st.sidebar.button("🔮 Predict App Cluster"):
    # Feature Engineering for Input Data
    engagement_score = (np.log1p(reviews) / np.log1p(installs + 1)) * rating

    # Standardizing User Input
    user_data = scaler.transform([[engagement_score, rating, reviews, installs]])

    # Predict Cluster
    predicted_cluster = kmeans.predict(user_data)[0]

    # Get Cluster Label
    predicted_label = cluster_labels[predicted_cluster]

    # Display Result
    st.sidebar.success(f"🔮 Your app belongs to: **{predicted_label}**")

# Display Cluster Count
st.subheader("📊 Cluster Distribution")
st.dataframe(df['Cluster Label'].value_counts())
