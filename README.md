# Telecom Churn Prediction 🚀📞

Predict customer churn for a telecom company using **machine learning** and deploy an interactive **Streamlit app** for real-time predictions.  
This project showcases skills in **Data Science, ML modeling, MLOps, and deployment with Docker**.

---

## 📝 Project Overview

The goal of this project is to **predict which customers are likely to churn**, helping the company make strategic decisions and improve retention.  
It includes **data exploration, predictive modeling, and an interactive web application**.

---


## 🛠 Technologies & Tools

- **Python:** pandas, numpy, scikit-learn, XGBoost, joblib  
- **Streamlit:** interactive web application  
- **Docker:** containerization for deployment  
- **Jupyter Notebook:** exploratory data analysis and experiments  
- **Git & GitHub:** version control  

---

## 🔍 Workflow

1. **Data Loading:** Load CSV files from `data/raw`.  
2. **Preprocessing:** Clean and transform data, save to `data/processed`.  
3. **Exploratory Data Analysis (EDA):** Visualize trends and patterns in notebooks.  
4. **Modeling:** Train multiple ML models and select the best one based on **Recall** and **ROC-AUC**.  
5. **Deployment:**  
   - Streamlit app (`app/app.py`) accepts an Excel file with customer data.  
   - Computes churn probability using the best model (`models/best_model.pkl`).  
   - Dockerfile enables easy deployment in any environment.

---

## 🤖 Models & Results

- Tested models: Logistic Regression, Random Forest, SVM and XGBoost  
- **Best model:** XGBoost  
  - **Recall:** 83.21%  
  - **ROC-AUC:** 88.91%  
- Most important features:
  - `Tenure in Months`
  - `Monthly Charge`
  - `Avg Monthly GB Download`
  - `Contract`
  - `Paperless Billing`
  - `Payment Method`
  - `Unlimited Data`

---


## 📫 Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/%C3%A1ngel-j-g%C3%B3mez-alonso-6b176b215/)





