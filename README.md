# CreditWiseAI - Intelligent Creditworthiness Prediction App

##  Project Overview

**CreditWiseAI** is a robust, ML-powered Streamlit web application Project that predicts the **creditworthiness** of individuals using advanced ensemble modeling techniques and real-world financial data. This solution integrates preprocessing, model tuning, cost-sensitive evaluation, and SHAP interpretability into an intuitive user interface for quick and reliable assessments.


> Built as part of an internship task to demonstrate end-to-end machine learning pipeline and model explainability.

---

##  Key Features

-  **Random Forest Classification** and **XGBoost** for improved prediction performance
-  **Data Preprocessing** and feature engineering
-  **Visualizations** to understand feature impact and distributions
-  **Model Evaluation** using classification metrics & ROC AUC
-  **SHAP Explainability** to understand model decisions
-  **Streamlit**-based web interface for real-time predictions

---

##  Problem Statement

> **Goal**: Predict whether a customer is **creditworthy** based on their financial and behavioral attributes using a machine learning classification approach.

---

##  Dataset Overview

The dataset consists of several financial features such as:

- `Age`, `Income`, `Employment Type`
- `Number of Open Credit Lines`
- `Credit History Length`, `Loan Purpose`
- `Overdue Count`, `Loan Amount`, etc.

> *(Dataset used is cleaned and anonymized/synthetic for academic use.)*

---

##  Tech Stack

| Tool/Library     | Purpose                            |
|------------------|------------------------------------|
| **Python**       | Programming language               |
| **Pandas, NumPy**| Data cleaning & transformation     |
| **Matplotlib, Seaborn** | Visualization libraries      |
| **Imbalance**    | Balanced Dataset                   |
| **Scikit-learn** | ML algorithms and evaluation       |
| **SHAP**         | Model interpretability             |


---

##  Project Pipeline

1.  Data Loading & Cleaning  
2.  Feature Engineering & Preprocessing  (OneHoe Encoder, LabelEncoder, Column Transformer)
3.  Train-Test Splitting  
4.  Model Training (Random Forest)  
5.  Performance Evaluation (Accuracy, ROC-AUC, Confusion Matrix)
6.  Hyperparameter Tuning (GridSearchCV)
7.  Interpret Results using SHAP plots  

---

##  Model Performance

- **Accuracy**: *0.7350*
- **ROC-AUC Score**: *0.7882*
- **Confusion Matrix**:
  
                 Predicted
               |  1   |  0
             -------------------
               | 31   | 29  |   0
             -------------------    Actual
               | 118  | 22  |   1

- **Top Features**: Duration_Months , Credit_Amount

---

##  How to Run This Project

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/CreditWiseAI.git
    cd CreditWiseAI
    ```
2. Open the notebook:
    ```bash
    jupyter notebook CreditWiseAI.ipynb
    ```
3. Run Streamlit App:
   ```bash
    streamlit run CreditWiseAI_app.py
    ```

---

##  Future Improvements

-  Containizer with **Docker**
-  Implement Model Tracking and Monitoring 
-  Add unit tests and ML pipeline CI/CD

---

##  About Me

I'm **Raghav Gaur**, an aspiring **Data Analyst/Machine Learning Engineer** with a passion for real-world machine learning applications.

-- Connect with me:  
[LinkedIn](https://linkedin.com/in/raghav--gaur)  

---

##  Show Your Support

If you found this project helpful or interesting, don’t forget to:

- ⭐ Star this repo  
- 🔄 Fork it  
- 🐛 Open issues or suggestions

---

> *This project was built as part of a data science internship assignment focused on ensemble methods for credit prediction.*
