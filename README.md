# Predicting-High-Electricity-Production-Using-Global-Energy-Source-Trends-Project
Predicting High Electricity Production Using Global Energy Source Trends Project
## 📊 Project Overview
- **Objective:** Predict whether a country’s **electricity production level** is **High (1)** or **Low (0)** based on features such as energy source, renewability, and development status.  
  > ⚠️ Note: The target variable `production_level` was derived from `electricity_production_gwh` by comparing each row to the **median value**. This binary transformation ensures that the model predicts high vs. low production **without leaking raw production values**.  
- **Dataset:** Global Monthly Electricity Production (4,721 rows × 23 columns).  
- **Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
⚙️ Workflow

Data Cleaning: Removed nulls, dropped irrelevant columns, handled missing regions.

EDA: Distribution plots, correlations, and key variable insights.

Feature Engineering: Encoded categorical variables (One-Hot, Label, Frequency, and Ordinal Encoding).

Model Building: Trained and evaluated four models using production_level as the target:

Logistic Regression

Random Forest 🌟 (Best Model – 94% accuracy)

SVM

KNN

Model Evaluation:

Confusion Matrix

Classification Report

Feature Importance

Prediction Pipeline: Created for real-time predictions on new data (full_pipeline).
