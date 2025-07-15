import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
import io


# Suppress warnings for a cleaner UI
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Creditworthiness Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Custom Cost Scorer (from notebook) ---
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

# --- 2. Data Loading and Preprocessing (Cached to run once) ---
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

    # Convert target variable 'class' to 0 and 1 (1: good credit, 0: bad credit)
    df['class'] = df['class'].map({1: 1, 2: 0})

    # Feature Engineering: 'Credit_Per_Duration'
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

    # Ge't feature names after one-hot encoding
    onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = list(numerical_cols) + list(onehot_feature_names)

    # Convert processed arrays back to DataFrames
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index) 

    # Handle imbalanced data with SMOTE 
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed_df, y_train)

    # --- Model Training (Random Forest Classifier - tuned from notebook) ---
    # Using the best parameters found in the notebook:
    # {'bootstrap': True, 'class_weight': None, 'max_depth': None, 'max_features': 'log2',
    #  'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
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

    return preprocessor, best_model, X.columns.tolist() # Return original columns for input fields

# Load preprocessor and model
preprocessor, model, original_columns = load_and_preprocess_data()

# --- 3. Streamlit App Interface ---
# Custom CSS for a more impactful interface
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .st-emotion-cache-18ni7ap { /* Header */
        font-size: 3.5em;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5em;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-h4xjwx { /* Subheader */
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
    .st-emotion-cache-16txtv3 { /* Sidebar header */
        font-size: 1.5em;
        color: #2c3e50;
        font-weight: 600;
    }
    .st-emotion-cache-10qtn7 { /* Expander header */
        font-size: 1.1em;
        font-weight: 600;
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Creditworthiness Assessment System")
st.subheader("Predicting Credit Risk with Precision")

st.write("""
Welcome to the Creditworthiness Assessment System. This tool leverages a sophisticated Machine Learning model trained to predict the credit risk of an applicant.
Please enter the applicant's details below to get an instant assessment.
""")

# --- Input Fields for User ---
st.sidebar.header("Applicant Details")

# Define input fields based on the mapped columns and their types/categories
input_data = {}

# Helper function for selectbox options
def get_selectbox_options(attribute_code_prefix, mapping):
    return {v: k for k, v in mapping.items() if k.startswith(attribute_code_prefix)}

# Define mappings for categorical features
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
    'female : single': 'female : single' # Not Exists in the data
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

with st.sidebar.form("input_form"):
    st.subheader("Financial Information")
    input_data['Checking_Account_Status'] = st.selectbox("Checking Account Status", list(CHECKING_ACCOUNT_STATUS_MAP.values()), format_func=lambda x: list(CHECKING_ACCOUNT_STATUS_MAP.keys())[list(CHECKING_ACCOUNT_STATUS_MAP.values()).index(x)])
    input_data['Duration_Months'] = st.number_input("Duration of Credit (Months)", min_value=4, max_value=72, value=12, step=1)
    input_data['Credit_History'] = st.selectbox("Credit History", list(CREDIT_HISTORY_MAP.values()), format_func=lambda x: list(CREDIT_HISTORY_MAP.keys())[list(CREDIT_HISTORY_MAP.values()).index(x)])
    input_data['Purpose'] = st.selectbox("Purpose of Credit", list(PURPOSE_MAP.values()), format_func=lambda x: list(PURPOSE_MAP.keys())[list(PURPOSE_MAP.values()).index(x)])
    input_data['Credit_Amount'] = st.number_input("Credit Amount (Rs.)", min_value=1000, max_value=100000, value=10000, step=100)
    input_data['Savings_Account_Bonds'] = st.selectbox("Savings Account/Bonds", list(SAVINGS_ACCOUNT_BONDS_MAP.values()), format_func=lambda x: list(SAVINGS_ACCOUNT_BONDS_MAP.keys())[list(SAVINGS_ACCOUNT_BONDS_MAP.values()).index(x)])
    input_data['Installment_Rate_Income'] = st.slider("Installment Rate in % of Disposable Income", min_value=0, max_value=15, value=5)
    input_data['Other_Installment_Plans'] = st.selectbox("Other Installment Plans", list(OTHER_INSTALLMENT_PLANS_MAP.values()), format_func=lambda x: list(OTHER_INSTALLMENT_PLANS_MAP.keys())[list(OTHER_INSTALLMENT_PLANS_MAP.values()).index(x)])
    input_data['Number_Existing_Credits'] = st.slider("Number of Existing Credits at this Bank", min_value=0, max_value=5, value=1)

    st.subheader("Personal & Demographic Information")
    input_data['Personal_Status_Gender'] = st.selectbox("Personal Status & Gender", list(PERSONAL_STATUS_GENDER_MAP.values()), format_func=lambda x: list(PERSONAL_STATUS_GENDER_MAP.keys())[list(PERSONAL_STATUS_GENDER_MAP.values()).index(x)])
    input_data['Employment_Duration'] = st.selectbox("Employment Duration", list(EMPLOYMENT_DURATION_MAP.values()), format_func=lambda x: list(EMPLOYMENT_DURATION_MAP.keys())[list(EMPLOYMENT_DURATION_MAP.values()).index(x)])
    input_data['Residence_Duration'] = st.slider("Residence Duration (Years at current address)", min_value=1, max_value=4, value=2)
    input_data['Property'] = st.selectbox("Property Type", list(PROPERTY_MAP.values()), format_func=lambda x: list(PROPERTY_MAP.keys())[list(PROPERTY_MAP.values()).index(x)])
    input_data['Age'] = st.number_input("Age (Years)", min_value=18, max_value=75, value=21, step=1)
    input_data['Housing'] = st.selectbox("Housing Type", list(HOUSING_MAP.values()), format_func=lambda x: list(HOUSING_MAP.keys())[list(HOUSING_MAP.values()).index(x)])
    input_data['Job'] = st.selectbox("Job Type", list(JOB_MAP.values()), format_func=lambda x: list(JOB_MAP.keys())[list(JOB_MAP.values()).index(x)])
    input_data['Number_People_Maintenance'] = st.radio("Number of people being liable to provide maintenance for", [1, 2], index=0)
    input_data['Telephone'] = st.radio("Has Telephone (registered in customer's name)?", list(TELEPHONE_MAP.values()), index=0, format_func=lambda x: list(TELEPHONE_MAP.keys())[list(TELEPHONE_MAP.values()).index(x)])
    input_data['Foreign_Worker'] = st.radio("Is Foreign Worker?", list(FOREIGN_WORKER_MAP.values()), index=0, format_func=lambda x: list(FOREIGN_WORKER_MAP.keys())[list(FOREIGN_WORKER_MAP.values()).index(x)])
    input_data['Other_Debtors_Guarantors'] = st.selectbox("Other Debtors / Guarantors", list(OTHER_DEBTORS_GUARANTORS_MAP.values()), format_func=lambda x: list(OTHER_DEBTORS_GUARANTORS_MAP.keys())[list(OTHER_DEBTORS_GUARANTORS_MAP.values()).index(x)])

    submitted = st.form_submit_button("Assess Creditworthiness")

if submitted:
    # Convert input data to a DataFrame
    processed_input = {}
    for key, value in input_data.items():
        if key == 'Checking_Account_Status':
            processed_input[key] = next(k for k, v in CHECKING_ACCOUNT_STATUS_MAP.items() if v == value)
        elif key == 'Credit_History':
            processed_input[key] = next(k for k, v in CREDIT_HISTORY_MAP.items() if v == value)
        elif key == 'Purpose':
            processed_input[key] = next(k for k, v in PURPOSE_MAP.items() if v == value)
        elif key == 'Savings_Account_Bonds':
            processed_input[key] = next(k for k, v in SAVINGS_ACCOUNT_BONDS_MAP.items() if v == value)
        elif key == 'Employment_Duration':
            processed_input[key] = next(k for k, v in EMPLOYMENT_DURATION_MAP.items() if v == value)
        elif key == 'Personal_Status_Gender':
            processed_input[key] = next(k for k, v in PERSONAL_STATUS_GENDER_MAP.items() if v == value)
        elif key == 'Other_Debtors_Guarantors':
            processed_input[key] = next(k for k, v in OTHER_DEBTORS_GUARANTORS_MAP.items() if v == value)
        elif key == 'Property':
            processed_input[key] = next(k for k, v in PROPERTY_MAP.items() if v == value)
        elif key == 'Other_Installment_Plans':
            processed_input[key] = next(k for k, v in OTHER_INSTALLMENT_PLANS_MAP.items() if v == value)
        elif key == 'Housing':
            processed_input[key] = next(k for k, v in HOUSING_MAP.items() if v == value)
        elif key == 'Job':
            processed_input[key] = next(k for k, v in JOB_MAP.items() if v == value)
        elif key == 'Telephone':
            processed_input[key] = next(k for k, v in TELEPHONE_MAP.items() if v == value)
        elif key == 'Foreign_Worker':
            processed_input[key] = next(k for k, v in FOREIGN_WORKER_MAP.items() if v == value)
        else:
            processed_input[key] = value

    input_df = pd.DataFrame([processed_input])

    # Ensure the input DataFrame has all original columns in the correct order
    # Create a dummy DataFrame with all original columns from X_train for consistent 
    dummy_df = pd.DataFrame(columns=original_columns)
    input_df = pd.concat([dummy_df, input_df], ignore_index=True).iloc[0:1] # Take only the first row (user input)

    # Re-apply feature engineering: Credit_Per_Duration
    input_df['Credit_Per_Duration'] = input_df['Credit_Amount'] / input_df['Duration_Months']

    # Preprocess the input data
    # We need to ensure the columns are in the same order as when the preprocessor was fitted.
    input_processed = preprocessor.transform(input_df)

    # Make prediction
    prediction = model.predict(input_processed)[0]
    prediction_proba = model.predict_proba(input_processed)[0]

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

    st.markdown("---")
    st.info("Disclaimer: This is a predictive model. All credit decisions should be made by qualified professionals.")

    st.download_button(
        label="Download Report",
        data=input_df.to_csv(index=False),
        file_name='credit_report.csv',
        mime='text/csv'
    )

    if st.sidebar.button("Reset All Fields"):
        st.experimental_rerun()
