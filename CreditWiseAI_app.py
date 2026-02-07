import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import shap
import datetime

# ML/DL Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input

warnings.filterwarnings("ignore")


# 1. UI MAPPINGS

MAPPINGS = {
    "Checking": {"A11": "< 0 RS", "A12": "0 - 200 RS", "A13": "> 200 RS", "A14": "None"},
    "History": {"A30": "No Credits", "A31": "Paid Back All", "A32": "Existing Paid", "A33": "Delay Past", "A34": "Critical"},
    "Savings": {"A61": "< 100 RS", "A62": "100-500 RS", "A63": "500-1000 RS", "A64": "> 1000 RS", "A65": "None/Unknown"},
    "Employment": {"A71": "Unemployed", "A72": "< 1 Year", "A73": "1-4 Years", "A74": "4-7 Years", "A75": ">= 7 Years"},
    "Housing": {"A151": "Rent", "A152": "Own", "A153": "Free"}
}


# 2. CORE LOGIC (Cached)

@st.cache_resource
def load_and_train():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    cols = [f"A{i}" for i in range(1, 21)] + ["class"]
    df = pd.read_csv(url, sep=" ", header=None, names=cols)
    df["class"] = df["class"].map({1: 1, 2: 0})
    df["Credit_Per_Duration"] = df["A5"] / df["A2"]

    X = df.drop("class", axis=1)
    y = df["class"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), X.select_dtypes(exclude="object").columns),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), X.select_dtypes("object").columns)
    ])

    X_p = preprocessor.fit_transform(X)
    feature_names = list(X.select_dtypes(exclude="object").columns) + \
                    list(preprocessor.named_transformers_["cat"].get_feature_names_out())

    # LSTM Feature Extractor
    inp = Input(shape=(1, X_p.shape[1]))
    x = LSTM(16)(inp)
    lstm_model = Model(inp, x) # Corrected name match
    
    X_hybrid = np.hstack([X_p, lstm_model.predict(X_p.reshape(X_p.shape[0], 1, -1))])
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_hybrid, y)

    return preprocessor, model, lstm_model, feature_names, X.head(1)

preprocessor, model, lstm_model, base_features, template_df = load_and_train()


# 3. VISUALIZATION FUNCTIONS

def draw_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, 40], 'color': "#FF4B4B"},
                {'range': [40, 70], 'color': "#FFAA00"},
                {'range': [70, 100], 'color': "#00CC96"}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig



# 4. STREAMLIT APP UI
st.set_page_config(page_title="CreditWise AI", layout="wide")
st.title("💳 CreditWiseAI | Business Decision Support")

with st.sidebar:
    st.header("📋 Applicant Profile")
    with st.form("input_form"):
        age = st.slider("Age", 18, 75, 35)
        amount = st.number_input("Requested Principal (RS)", 500, 20000, 3000)
        duration = st.slider("Duration (Months)", 6, 72, 24)
        
        # NEW: Interest Input
        interest_rate = st.slider("Annual Interest Rate (%)", 1.0, 25.0, 5.0)
        
        st.markdown("---")
        chk = st.selectbox("Checking Status", options=list(MAPPINGS["Checking"].keys()), format_func=lambda x: MAPPINGS["Checking"][x])
        sav = st.selectbox("Savings Status", options=list(MAPPINGS["Savings"].keys()), format_func=lambda x: MAPPINGS["Savings"][x])
        hist = st.selectbox("Credit History", options=list(MAPPINGS["History"].keys()), format_func=lambda x: MAPPINGS["History"][x])
        emp = st.selectbox("Work Seniority", options=list(MAPPINGS["Employment"].keys()), format_func=lambda x: MAPPINGS["Employment"][x])
        
        submit = st.form_submit_button("Generate Assessment")


# 5. CALCULATIONS & PREDICTION

if submit:
    # Math: Total Repayment Calculation
    # Simple Interest: A = P(1 + rt) where r is decimal and t is years
    total_repayment = amount * (1 + (interest_rate/100 * (duration/12)))
    monthly_installment = total_repayment / duration

    # Prepare Data using Direct Assignment to fix scalar index error
    input_df = template_df.copy()
    input_df.at[input_df.index[0], "A2"] = duration
    input_df.at[input_df.index[0], "A5"] = amount
    input_df.at[input_df.index[0], "A13"] = age
    input_df.at[input_df.index[0], "A1"] = chk
    input_df.at[input_df.index[0], "A6"] = sav
    input_df.at[input_df.index[0], "A3"] = hist
    input_df.at[input_df.index[0], "A7"] = emp
    input_df.at[input_df.index[0], "Credit_Per_Duration"] = amount / duration

    # Prediction Pipeline
    X_p = preprocessor.transform(input_df)
    X_lstm = lstm_model.predict(X_p.reshape(1, 1, -1))
    X_hybrid = np.hstack([X_p, X_lstm])
    prob_good = model.predict_proba(X_hybrid)[0][1]

    # Display Visuals
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("Total Repayment", f"{total_repayment:,.2f} RS")
        st.metric("Monthly Payment", f"{monthly_installment:,.2f} RS")

    with col2:
        st.plotly_chart(draw_gauge(prob_good), use_container_width=True)
        if prob_good > 0.6: st.success(" LOW RISK")
        elif prob_good > 0.4: st.warning("🟡 MEDIUM RISK")
        else: st.error(" HIGH RISK")

    with col3:
        st.subheader("Analysis")
        # Visualizing the influence of key factors
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_hybrid)
        
        # Simplify names for non-techs
        display_names = base_features + [f"Pattern {i}" for i in range(16)]
        pred_class = 1 if prob_good > 0.5 else 0
        impacts = shap_values[pred_class][0] if isinstance(shap_values, list) else shap_values[0, :, pred_class]
        
        influence_df = pd.DataFrame({"Factor": display_names, "Impact": impacts})
        influence_df = influence_df.sort_values(by="Impact", key=abs, ascending=False).head(5)
        
        fig_bar = px.bar(influence_df, x="Impact", y="Factor", orientation='h',
                         title="Top Decision Drivers",
                         color=influence_df["Impact"] > 0,
                         color_discrete_map={True: "#00CC96", False: "#FF4B4B"})
        fig_bar.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.caption(f"Assessment generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
