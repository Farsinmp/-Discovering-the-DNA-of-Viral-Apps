# Discovering the DNA of Viral Apps

Uncover what makes an app go viral on the Google Play Store using data science techniques like clustering, PCA, and interactive visualizations.

#🔍 Overview

This project analyzes the Google Play Store dataset to extract key features that contribute to an app's virality. By engineering features such as Virality Score and Engagement Score, and applying clustering techniques, we categorize apps into various virality levels. The insights are presented through a Streamlit web app and a Power BI dashboard.

#🎯 Objectives

Identify the features that drive app virality.

Create standardized metrics for Virality and Engagement.

Use K-Means Clustering to group apps based on virality traits.

Apply PCA for dimensionality reduction and visualization.

Build a Streamlit app to interactively explore insights.

Develop a Power BI dashboard for high-level business understanding.

#🛠️ Tech Stack

Tool	Purpose
Python	Core data processing & ML
Pandas, NumPy	Data manipulation
Matplotlib, Seaborn	Visualization
Scikit-learn	Clustering, PCA
Streamlit	Web app deployment
Power BI	Dashboard and data storytelling
Jupyter Notebook	Development environment
#🧪 Methodology
Data Cleaning: Removed duplicates, handled missing values, and converted data types.

Feature Engineering:

Created Virality Score from installs, ratings, and reviews.

Designed Engagement Score based on app usage features.

Standardization: Scaled numerical features for clustering.

K-Means Clustering:

Used Elbow Method to find optimal clusters.

Categorized apps into virality levels.

Dimensionality Reduction:

Applied PCA to visualize clusters in 2D.

Deployment:

Developed an interactive Streamlit dashboard.

Created Power BI reports for strategic insights.

#📊 Power BI Dashboard
Power BI report available in the /dashboard/ folder. Import into Power BI Desktop to explore:

Virality vs. Engagement trends

Free vs. Paid apps analysis

Cluster-wise app distribution

Top-performing apps by category
#📌 Future Enhancements
Add model to predict virality score for new apps

Integrate real-time Google Play data (via scraping/API)

Deploy on cloud (Heroku/Streamlit Cloud)

#🙌 Acknowledgements
Dataset: Kaggle - Google Play Store Apps
Inspiration: Viral growth patterns and app marketing strategies.

#🧑‍💻 Author
Muhammed Farsin MP
Machine Learning Developer | Data Enthusiast | Passionate about Trading
📍 Kerala, India
