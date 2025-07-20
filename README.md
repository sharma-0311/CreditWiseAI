# CreditWiseAI - Intelligent Creditworthiness Prediction App

##  Project Overview

**CreditWiseAI** is a full-fledged, end-to-end credit risk prediction platform built with **Streamlit**, **scikit-learn**, and **MySQL**, designed for real-world use by financial institutions. This AI-powered app predicts whether a customer is creditworthy based on financial and behavioral attributes, with advanced ML techniques, database logging, and SHAP explainability.


> Built as part of an internship task to demonstrate end-to-end machine learning pipeline and model explainability.

---

##  Key Features

-  **Real-time Streamlit** web interface 
-  **Random Forest Classification** and **XGBoost** for improved prediction performance
-  Fully integrated **MySQL** backend for input logging & audit trail 
-  **Data Preprocessing** and feature engineering
-  **Visualizations** to understand feature impact and distributions
-  **Model Evaluation** using classification metrics & ROC AUC
-  **SHAP Explainability** to understand model decisions
-  No sensitive info used — synthetic/academic dataset  

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
| **MySQL + Connector** | Backend database integration  |


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

## 🔗 Project Architecture

```text
[User Input] 
    ↓
[Streamlit UI]
    ↓
[Data Preprocessing → Model Prediction]
    ↓                         ↓
[MySQL Logging]        [SHAP Explanation + PDF Report]
    ↓                         ↓
[Download Report]        [Optional RAG Chatbot Integration]

```
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

MySQL Integration

The project logs:
- Applicant details (optional name, age, loan info)
- Prediction results and confidence scores

SQL tables:
```
-- applicant_inputs
-- prediction_logs
```
View the schema in `schema.sql`

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
-  LangChain + RAG integration for credit policy Q&A
-  Deploy on Streamlit Cloud or Render

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

## Acknowledgments

Built during a **Data Science Internship** @ Celebal Technologies
Special thanks to mentors, peers, and the Celebal community
