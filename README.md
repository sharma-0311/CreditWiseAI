# Credit Analyzing System using Random Forest

##  Project Overview

The **Credit Analyzing System** is a machine learning project designed to predict the **creditworthiness of individuals** using financial and demographic data. It utilizes a **Random Forest Classifier**, **XGBoost** and **LogisticRegression** for accurate and reliable predictions — making it suitable for real-world applications in banking and finance.

> Built as part of an internship task to demonstrate end-to-end machine learning pipeline and model explainability.

---

##  Key Features

-  **Random Forest Classification** and **XGBoost** for improved prediction performance
-  **Data Preprocessing** and feature engineering
-  **Visualizations** to understand feature impact and distributions
-  **Model Evaluation** using classification metrics & ROC AUC
-  **SHAP Explainability** to understand model decisions

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
  
  | 29 | 31 |
  | 22 |118 |
  
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

---

##  Future Improvements

-  Deploy using **Streamlit** or **Flask**
-  Containizer with **Docker**
-  Implement Model Monitoring
-  Add unit tests and ML pipeline CI/CD

---

##  About Me

I'm **Raghav Gaur**, an aspiring **Data Analyst/Machine Learning Engineer** with a passion for real-world machine learning applications.

📫 Connect with me:  
[LinkedIn](https://linkedin.com/in/raghav--gaur)  

---

##  Show Your Support

If you found this project helpful or interesting, don’t forget to:

- ⭐ Star this repo  
- 🔄 Fork it  
- 🐛 Open issues or suggestions

---

> *This project was built as part of a data science internship assignment focused on ensemble methods for credit prediction.*
