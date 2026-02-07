# CreditWiseAI - Intelligent Creditworthiness Prediction App

##  Project Overview

**CreditWiseAI** is a production-grade, end-to-end credit risk **assessment platform** built using **Streamlit, scikit-learn, TensorFlow (LSTM)**, and **Firebase**.

The system predicts whether an applicant is creditworthy or high-risk by combining:

- Traditional machine learning (Random Forest / XGBoost)
- Deep learning (LSTM) for behavioral pattern extraction
- Explainable AI (SHAP) for transparency
- RAG-powered GenAI chatbot for credit knowledge assistance

> Built as an internship / PPO-level project to demonstrate real-world ML system design, explainability, and deployment readiness.

---

##  Key Features

-  **Real-time Streamlit** web interface 
-  **Hybrid Model Architecture**
-- **LSTM** for temporal/behavioral feature learning
-- **Random Forest / XGBoost** for final decision making
-  **Data Preprocessing** and feature engineering
-  **Visualizations** to understand feature impact and distributions
-  **Model Evaluation** using classification metrics & ROC AUC
-  **SHAP Explainability** to understand model decisions
-  **Data Logging**: Input and predictions logged in MySQL
-  **AI Chatbot**: LangChain-powered Q&A chatbot (GPT-3.5 / Gemini)
-  **Modular Codebase**: Clean separation of UI, ML, database, and RAG pipeline
-  **Fast & Lightweight**: Optimized for performance and responsiveness
-  No sensitive info used — synthetic/academic dataset

---

##  Problem Statement

> **Goal**: Predict whether a customer is **creditworthy** based on their financial and behavioral attributes using a machine learning approach.

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
| **Matplotlib, Seaborn** | Visualization libraries     |
| **Imbalearn**    | Balanced Dataset                   |
| **Scikit-learn** | ML algorithms and evaluation       |
| **SHAP**         | Model interpretability             |
| **MySQL + Connector** | Backend database integration  |
| **LangChain + FAISS** | RAG Vector DB for Chatbot     |
| **OpenAI / Gemini API** | LLM-based Credit Q&A Assistant|


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

##  Project Architecture

```text
[User Input] 
    ↓
[Streamlit UI]
    ↓
[Data Preprocessing → Model Prediction]
    ↓                         ↓
[MySQL Logging]        [SHAP Explanation + PDF Report]
    ↓                         ↓
[Download Report]        [RAG Chatbot Integration]

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
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Setup your .env
   ```bash
   touch .env
   ```
5. Add your keys inside
   ```bash
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_google_key
   ```
6. Run Vector DB Loader
   ```bash
   python run_loader.py
   ```
7. Run Streamlit App:
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
-  LangChain Agent Tools for advanced reasoning

---

##  About Me

I'm **Raghav Gaur**, an aspiring **Data Analyst/Machine Learning Engineer** with a passion for real-world machine learning applications.

-- Connect with me:  
[LinkedIn](https://linkedin.com/in/raghav--gaur)  
[Email] : rgour6350@gmail.com

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
