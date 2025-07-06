**CreditWiseAI** an **ml-credit-evaluator**

# Creditworthiness Prediction using Random Forest

This project is a machine learning solution that predicts the **creditworthiness of individuals** based on financial and personal attributes. The goal is to classify whether a person is likely to be a **good or bad credit risk**, using a Random Forest ensemble method.

---

## Objective

To fulfill the internship assignment of:
> **"Using a Random Forest ensemble method to predict creditworthiness of individuals based on various financial attributes."**

This involves:
- Preprocessing raw financial data
- Training a predictive model
- Evaluating model performance
- Interpreting predictions using explainable AI (SHAP)

---

## Dataset

**Source**: [UCI German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))  
**Attributes**: 20 input features + 1 binary target (`good` or `bad` credit)

---

## Project Workflow

1. **Data Acquisition**  
   - Retrieved using the `ucimlrepo` Python library directly from UCI.

2. **Data Preprocessing**  
   - Renamed ambiguous attribute names for clarity  
   - Categorical encoding via `LabelEncoder`  
   - Target variable converted from text to binary labels

3. **Exploratory Data Analysis**  
   - Distribution plots for target variable  
   - Correlation heatmap for feature relationships  
   - KDE plots and pairplots to understand data distributions

4. **Modeling**  
   - Used `RandomForestClassifier` from `sklearn`  
   - 80/20 stratified train-test split  
   - Evaluated using:
     - Accuracy Score
     - Confusion Matrix
     - Classification Report

5. **Model Explainability**  
   - Integrated `SHAP` (SHapley Additive Explanations)  
   - Force plots for instance-level interpretation  
   - Identified key contributing features

---

## Results

- **Model Accuracy**: ~`1.0`
- **Top Influential Features**:
  - `Credit_Amount`
  - `Duration_Months`
  - `Age`
  - `Checking_Account_Status`
  - `Savings_Account`

---

## Conclusion

This project successfully demonstrates the application of ensemble learning for credit scoring. By combining **Random Forest classification** with **SHAP explainability**, we not only achieved high predictive performance but also gained **insight into feature contributions**, making the model more transparent and trustworthy — an essential aspect in the financial domain.

---

## Tech Stack

- Python 3.12
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- SHAP
- ucimlrepo

---

## Author

> **Raghav Gaur**  
> Data Science Intern | Aspiring Machine Learning Engineer 
> [Bareilly, India]  

---

