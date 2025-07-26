import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sql_utils import insert_applicant, insert_prediction
import warnings
from dotenv import load_dotenv
import io
import os
import traceback

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Creditworthiness Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize session state for navigation ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

# --- Custom Cost Scorer (from notebook) ---
def custom_cost_scorer(y_true, y_pred):
    """
    Custom scorer based on the problem's misclassification costs:
    False Positive (FP): 5 units
    False Negative (FN): 1 unit
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = (fp * 5) + (fn * 1)
    return -cost

cost_scorer = make_scorer(custom_cost_scorer, greater_is_better=False)

# --- Data Loading and Preprocessing (Cached to run once) ---
@st.cache_resource
def load_and_preprocess_data():
    from ucimlrepo import fetch_ucirepo

    statlog_german_credit_data = fetch_ucirepo(id=144)
    X_raw = statlog_german_credit_data.data.features
    y_raw = statlog_german_credit_data.data.targets

    df = pd.concat([X_raw, y_raw], axis=1)

    # Map Attribute# to meaningful column names (from notebook)
    column_mapping = {
        'Attribute1': 'Checking_Account_Status',
        'Attribute2': 'Duration_Months',
        'Attribute3': 'Credit_History',
        'Attribute4': 'Purpose',
        'Attribute5': 'Credit_Amount',
        'Attribute6': 'Savings_Account_Bonds',
        'Attribute7': 'Employment_Duration',
        'Attribute8': 'Installment_Rate_Income',
        'Attribute9': 'Personal_Status_Gender',
        'Attribute10': 'Other_Debtors_Guarantors',
        'Attribute11': 'Residence_Duration',
        'Attribute12': 'Property',
        'Attribute13': 'Age',
        'Attribute14': 'Other_Installment_Plans',
        'Attribute15': 'Housing',
        'Attribute16': 'Number_Existing_Credits',
        'Attribute17': 'Job',
        'Attribute18': 'Number_People_Maintenance',
        'Attribute19': 'Telephone',
        'Attribute20': 'Foreign_Worker',
        'Attribute21': 'class'  # Target variable
    }
    df.rename(columns=column_mapping, inplace=True)
    df['class'] = df['class'].map({1: 1, 2: 0})
    df['Credit_Per_Duration'] = df['Credit_Amount'] / df['Duration_Months']

    X = df.drop('class', axis=1)
    y = df['class']

    # Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = list(numerical_cols) + list(onehot_feature_names)

    # Convert processed arrays back to DataFrames
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)

    # Handle imbalanced data with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed_df, y_train)

    # --- Model Training (Random Forest Classifier - tuned from notebook) ---
    best_model = RandomForestClassifier(
        bootstrap=True,
        class_weight=None,
        max_depth=None,
        max_features='log2',
        min_samples_leaf=2,
        min_samples_split=2,
        n_estimators=100,
        random_state=42
    )

    best_model.fit(X_train_resampled, y_train_resampled)

    return preprocessor, best_model, X.columns.tolist()

# Load preprocessor and model
preprocessor, model, original_columns = load_and_preprocess_data()

# --- Custom CSS for a more impactful interface ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        font-size: 3.5em;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5em;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h2 {
        font-size: 1.8em;
        color: #34495e;
        text-align: center;
        margin-bottom: 2em;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        font-size: 1.2em;
        padding: 0.8em 2em;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #218838;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        margin-top: 30px;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: all 0.5s ease;
    }
    .prediction-good {
        background-color: #28a745; /* Green */
    }
    .prediction-bad {
        background-color: #dc3545; /* Red */
    }
    </style>
    """, unsafe_allow_html=True)


# --- Function to display prediction ---
def show_prediction_results(prediction, prediction_proba, applicant_id):
    st.markdown("---")
    st.header("Prediction Result")

    if prediction == 1:
        st.markdown(
            f"<div class='prediction-box prediction-good'>Creditworthy!</div>",
            unsafe_allow_html=True
        )
        st.success(f"The model predicts this applicant is **Creditworthy** with a confidence of **{prediction_proba[1]*100:.2f}%**.")
    else:
        st.markdown(
            f"<div class='prediction-box prediction-bad'>High Risk!</div>",
            unsafe_allow_html=True
        )
        st.error(f"The model predicts this applicant is **High Risk** with a confidence of **{prediction_proba[0]*100:.2f}%**.")
        st.warning("Further review or additional information may be required for this application.")

    insert_prediction(applicant_id, int(prediction), float(prediction_proba[prediction]))

    st.markdown("---")
    st.info("Disclaimer: This is a predictive model. All credit decisions should be made by qualified professionals.")

    st.download_button(
        label="Download Report",
        data=pd.DataFrame([input_data]).to_csv(index=False),
        file_name='credit_report.csv',
        mime='text/csv'
    )

# --- Placeholder for SHAP Explainability ---
def show_shap_explainability():
    st.title("🔍 SHAP Explainability")
    st.markdown("Model explanation using SHAP values.")
    st.warning("SHAP explanation feature is under development. This section will show how each input feature contributes to the creditworthiness prediction.")
    st.info("Stay tuned for detailed model insights!")
    st.image("https://shap.readthedocs.io/en/latest/_images/shap_summary_plot.png", caption="SHAP Summary Example (Placeholder)") # Added placeholder image
    # TODO: Integrate SHAP plots here. This would require:
    # 1. Calculating SHAP values for a given prediction or global feature importance.
    # 2. Using shap.plots to render the visualizations.

# --- RAG Q&A Chatbot Section ---
def show_chatbot():
    st.title("💬 Credit Chatbot (GenAI)")
    st.markdown("Ask anything related to credit scoring or CreditWiseAI.")

    try:
        from Chatbot.chat_handler import handle_chat_query, get_llm_model_name
        import os

        # Check for FAISS index existence
        if not os.path.exists("Chatbot/faiss_index"):
            st.error("⚠️ Chatbot knowledge base not found! Please run `python run_loader.py` in your terminal to create the vector store.")
            st.info("Make sure you have documents in `Chatbot/documents` for the loader to process.")
            return

        # Check for OpenAI or Google API key
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            st.error("Neither OpenAI nor Google API key found. Please set `OPENAI_API_KEY` or `GOOGLE_API_KEY` in your `.env` file or environment variables.")
            st.info("You can get OpenAI key from platform.openai.com/account/api-keys or Google Gemini key from aistudio.google.com/app/apikey")
            return

        # Display which LLM is being used
        llm_in_use = get_llm_model_name()
        if llm_in_use:
            st.sidebar.info(f"Chatbot powered by: **{llm_in_use}**")
        else:
            st.sidebar.warning("Chatbot LLM could not be identified or initialized.")

    except ImportError:
        st.error("Chatbot module (Chatbot/chat_handler.py) could not be loaded. Please ensure it exists and dependencies are installed.")
        st.code(traceback.format_exc(), language="python")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during chatbot initialization: {e}")
        st.code(traceback.format_exc(), language="python")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask anything about credit, loans, or CreditWiseAI...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("ai"):
            with st.spinner("CreditWiseAI is thinking..."):
                try:
                    response = handle_chat_query(user_query)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "ai", "content": response})
                except Exception as e:
                    error_message = f"An error occurred while getting a response: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "ai", "content": error_message})
                    st.code(traceback.format_exc(), language="python")


# --- Sidebar Navigation ---
st.sidebar.title("🔘 CreditWiseAI Navigation")
tab_titles = ["Home", "Predict", "SHAP Explain", "Credit Chatbot"]
selected = st.sidebar.radio(
    "Go to",
    tab_titles,
    index=tab_titles.index(st.session_state.active_tab),
    key="tab_radio"
)

# Update active_tab based on radio selection
st.session_state.active_tab = selected


# --- Content rendering based on active_tab ---

if st.session_state.active_tab == "Home":
    st.title(" Welcome to CreditWiseAI")
    st.subheader("Predicting Credit Risk with Precision")
    st.markdown("""
        <p style='font-size: 1.1em;'>
        This application provides a comprehensive platform for assessing creditworthiness, explaining model decisions, and answering your financial queries.
        </p>
        <p style='font-size: 1.1em;'>
        Navigate through the options using the sidebar:
        <ul>
            <li><b>Predict:</b> Enter applicant details and get an instant creditworthiness prediction.</li>
            <li><b>SHAP Explain:</b> Understand why a particular credit decision was made (feature coming soon!).</li>
            <li><b>Credit Chatbot:</b> Ask questions about credit, loans, and general financial literacy.</li>
        </ul>
        </p>
    """, unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1579621970795-87facc2f976d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1740&q=80", use_container_width=True, caption="Credit and Finance Analytics")


elif st.session_state.active_tab == "Predict":
    st.title(" Predict Creditworthiness")
    st.sidebar.header("Applicant Details")
    input_data = {} # Initialize input_data here

    # Define mappings for categorical features (these are used for mapping input to model's expected format)
    CHECKING_ACCOUNT_STATUS_MAP = {
        '< 0 DM': '< 0 DM',
        '0 <= ... < 200 DM': '0 <= ... < 200 DM',
        '>= 200 DM / salary assignments for at least 1 year': '>= 200 DM / salary assignments for at least 1 year',
        'no checking account': 'no checking account'
    }

    CREDIT_HISTORY_MAP = {
        'No credits taken': 'no credits taken/all credits paid back duly',
        'all credits paid': 'all credits at this bank paid back duly',
        'existing credits paid': 'existing credits paid back duly till now',
        'delay in paying off credits': 'delay in paying off in the past',
        'Other credits existing(other bank)': 'critical account/other credits existing (not at this bank)'
    }

    PURPOSE_MAP = {
        'Car(new)': 'car (new)',
        'Car(used)': 'car (used)',
        'Furniture/Equipment': 'furniture/equipment',
        'Electronics': 'radio/television',
        'Domestic Appliances': 'domestic appliances',
        'Repairs': 'repairs',
        'Education': 'education',
        'Retraining': 'retraining',
        'Business': 'business',
        'Others': 'others'
    }

    SAVINGS_ACCOUNT_BONDS_MAP = {
        '< 1000 Rs.': '< 100 DM',
        '1000 - 5000 Rs.': '100 <= ... < 500 DM',
        '5000 - 10000 Rs.': '500 <= ... < 1000 DM',
        '> 1000 Rs.': '>= 1000 DM',
        'no savings account': 'unknown/no savings account'
    }

    EMPLOYMENT_DURATION_MAP = {
        'unemployed': 'unemployed',
        '< 1 year': '< 1 year',
        '1 - 4 years': '1 <= ... < 4 years',
        '4 - 7 years': '4 <= ... < 7 years',
        '> 7 years': '>= 7 years'
    }

    PERSONAL_STATUS_GENDER_MAP = {
        'male : divorced/separated': 'male : divorced/separated',
        'female : divorced/separated/married': 'female : divorced/separated/married',
        'male : single': 'male : single',
        'male : married/widowed': 'male : married/widowed',
        'female : single': 'female : single'
    }

    OTHER_DEBTORS_GUARANTORS_MAP = {
        'none': 'none',
        'co-applicant': 'co-applicant',
        'guarantor': 'guarantor'
    }

    PROPERTY_MAP = {
        'real estate': 'real estate',
        'building society savings agreement/life insurance': 'building society savings agreement/life insurance',
        'car or other': 'car or other, not in attribute 20',
        'no property': 'unknown / no property'
    }

    OTHER_INSTALLMENT_PLANS_MAP = {
        'bank': 'bank',
        'stores': 'stores',
        'other': 'other',
        'none': 'none'
    }

    HOUSING_MAP = {
        'rent': 'rent',
        'own': 'own',
    }

    JOB_MAP = {
        'unemployed - non-resident': 'unemployed/unskilled - non-resident',
        'unemployed - resident': 'unskilled - resident',
        'skilled employee / official': 'skilled employee / official',
        'management/self-employed/highly qualified employee/officer': 'management/self-employed/highly qualified employee/officer'
    }

    TELEPHONE_MAP = {
        'none': 'none',
        'yes, registered under the customer’s name': 'yes, registered under the customer’s name'
    }

    FOREIGN_WORKER_MAP = {
        'yes': 'yes',
        'no': 'no'
    }

    with st.form("input_form"):
        applicant_name = st.text_input("Applicant Name (optional)", value="Anonymous")
        st.subheader("Financial Information")
        input_data = {
            'Checking_Account_Status': st.selectbox("Bank Balance Status", list(CHECKING_ACCOUNT_STATUS_MAP.values())),
            'Duration_Months': st.slider("Loan Tenure (in Months)", 4, 72, 12),
            'Credit_History': st.selectbox("Credit History", list(CREDIT_HISTORY_MAP.values())),
            'Purpose': st.selectbox("Purpose of Loan", list(PURPOSE_MAP.values())),
            'Credit_Amount': st.number_input("Loan Amount (in Rs.)", 1000, 100000, 5000),
            'Savings_Account_Bonds': st.selectbox("Savings Account Status", list(SAVINGS_ACCOUNT_BONDS_MAP.values())),
            'Employment_Duration': st.selectbox("Years of Employment", list(EMPLOYMENT_DURATION_MAP.values())),
            'Installment_Rate_Income': st.slider("EMI as % of Income", 1, 15, 4),
            'Personal_Status_Gender': st.selectbox("Marital Status & Gender", list(PERSONAL_STATUS_GENDER_MAP.values())),
            'Other_Debtors_Guarantors': st.selectbox("Guarantor Type", list(OTHER_DEBTORS_GUARANTORS_MAP.values())),
            'Residence_Duration': st.slider("Years at Current Residence", 1, 4, 2),
            'Property': st.selectbox("Type of Owned Property", list(PROPERTY_MAP.values())),
            'Age': st.slider("Age of Applicant", 18, 75, 30),
            'Other_Installment_Plans': st.selectbox("Other EMI Commitments", list(OTHER_INSTALLMENT_PLANS_MAP.values())),
            'Housing': st.selectbox("Living Status", list(HOUSING_MAP.values())),
            'Number_Existing_Credits': st.slider("Number of Previous Loans at this Bank", 0, 4, 1),
            'Job': st.selectbox("Job Profile", list(JOB_MAP.values())),
            'Number_People_Maintenance': st.radio("Number of Dependents", [1, 2]),
            'Telephone': st.radio("Telephone Available?", list(TELEPHONE_MAP.values())),
            'Foreign_Worker': st.radio("Is the applicant a foreign national?", list(FOREIGN_WORKER_MAP.values())),
        }
        submitted = st.form_submit_button("Assess Creditworthiness")

        # SQL Input Data mapping to the DB schema
        sql_input_data = {
            'name': applicant_name,
            'age': input_data['Age'],
            'credit_amount': input_data['Credit_Amount'],
            'employment_duration': input_data['Employment_Duration'],
            'savings_status': input_data['Savings_Account_Bonds'],
            'loan_purpose': input_data['Purpose'],
            'housing': input_data['Housing'],
            'number_of_dependents': input_data['Number_People_Maintenance']
        }

    if submitted:
        applicant_id = insert_applicant(sql_input_data)
        # Convert input data to a DataFrame for prediction
        processed_input_for_model = {}
        for key, value in input_data.items():
            # Apply reverse mapping to get the original attribute codes
            if key == 'Checking_Account_Status':
                processed_input_for_model[key] = next(k for k, v in CHECKING_ACCOUNT_STATUS_MAP.items() if v == value)
            elif key == 'Credit_History':
                processed_input_for_model[key] = next(k for k, v in CREDIT_HISTORY_MAP.items() if v == value)
            elif key == 'Purpose':
                processed_input_for_model[key] = next(k for k, v in PURPOSE_MAP.items() if v == value)
            elif key == 'Savings_Account_Bonds':
                processed_input_for_model[key] = next(k for k, v in SAVINGS_ACCOUNT_BONDS_MAP.items() if v == value)
            elif key == 'Employment_Duration':
                processed_input_for_model[key] = next(k for k, v in EMPLOYMENT_DURATION_MAP.items() if v == value)
            elif key == 'Personal_Status_Gender':
                processed_input_for_model[key] = next(k for k, v in PERSONAL_STATUS_GENDER_MAP.items() if v == value)
            elif key == 'Other_Debtors_Guarantors':
                processed_input_for_model[key] = next(k for k, v in OTHER_DEBTORS_GUARANTORS_MAP.items() if v == value)
            elif key == 'Property':
                processed_input_for_model[key] = next(k for k, v in PROPERTY_MAP.items() if v == value)
            elif key == 'Other_Installment_Plans':
                processed_input_for_model[key] = next(k for k, v in OTHER_INSTALLMENT_PLANS_MAP.items() if v == value)
            elif key == 'Housing':
                processed_input_for_model[key] = next(k for k, v in HOUSING_MAP.items() if v == value)
            elif key == 'Job':
                processed_input_for_model[key] = next(k for k, v in JOB_MAP.items() if v == value)
            elif key == 'Telephone':
                processed_input_for_model[key] = next(k for k, v in TELEPHONE_MAP.items() if v == value)
            elif key == 'Foreign_Worker':
                processed_input_for_model[key] = next(k for k, v in FOREIGN_WORKER_MAP.items() if v == value)
            else:
                processed_input_for_model[key] = value

        input_df = pd.DataFrame([processed_input_for_model])

        # Ensure the input DataFrame has all original columns in the correct order
        dummy_df = pd.DataFrame(columns=original_columns)
        input_df = pd.concat([dummy_df, input_df], ignore_index=True).iloc[0:1]
        input_df['Credit_Per_Duration'] = input_df['Credit_Amount'] / input_df['Duration_Months']
        input_processed = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(input_processed)[0]
        prediction_proba = model.predict_proba(input_processed)[0]

        show_prediction_results(prediction, prediction_proba, applicant_id)

        # Automatically redirect to SHAP Explain tab after prediction
        st.session_state.active_tab = "SHAP Explain"
        # st.experimental_rerun() # This is generally not needed here as state change will trigger rerun

    # The reset button should be placed where it makes sense, maybe inside the prediction results section
    if st.button("Reset All Fields"):
        st.session_state.active_tab = "Home"
        st.experimental_rerun() # Force rerun to show Home tab and reset form values


elif st.session_state.active_tab == "SHAP Explain":
    show_shap_explainability()

elif st.session_state.active_tab == "Credit Chatbot":
    show_chatbot()
