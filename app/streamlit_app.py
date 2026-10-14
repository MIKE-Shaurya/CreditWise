

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="CreditWise", page_icon="💳", layout="centered")


def api_get(path: str):
    """GET from the API; stop the page with a clear message if that fails."""
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=5)
    except requests.exceptions.RequestException:
        st.error(f"Can't reach the API at {API_URL}. Is it running?")
        st.stop()
    if resp.status_code == 503:
        st.warning(resp.json().get("detail", "The model isn't available yet."))
        st.stop()
    resp.raise_for_status()
    return resp.json()


st.title("CreditWise")
st.caption("Estimate how likely a loan application is to be approved.")

options = api_get("/options")
info = api_get("/model-info")

with st.sidebar:
    st.subheader("Model")
    st.write(info["model_name"])
    left, right = st.columns(2)
    left.metric("Accuracy", f"{info['metrics']['accuracy']:.1%}")
    right.metric("F1", f"{info['metrics']['f1']:.2f}")
    st.caption(f"Trained {info['trained_at']}")
    st.caption(f"API: {API_URL}")

with st.form("application"):
    st.subheader("Applicant")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", min_value=18, max_value=100, value=35)
    gender = c2.selectbox("Gender", options["Gender"])
    marital = c1.selectbox("Marital status", options["Marital_Status"])
    education = c2.selectbox("Education", options["Education_Level"])
    dependents = c1.number_input("Dependents", min_value=0, value=0)
    area = c2.selectbox("Property area", options["Property_Area"])

    st.subheader("Employment and income")
    c1, c2 = st.columns(2)
    employment = c1.selectbox("Employment status", options["Employment_Status"])
    employer = c2.selectbox("Employer category", options["Employer_Category"])
    income = c1.number_input("Applicant income (monthly)", min_value=0.0, value=5000.0, step=100.0)
    co_income = c2.number_input("Co-applicant income (monthly)", min_value=0.0, value=0.0, step=100.0)

    st.subheader("Credit and assets")
    c1, c2 = st.columns(2)
    credit_score = c1.number_input("Credit score", min_value=300, max_value=900, value=700)
    existing_loans = c2.number_input("Existing loans", min_value=0, value=0)
    dti = c1.number_input("DTI ratio", min_value=0.0, value=0.30, step=0.01, format="%.2f",
                          help="Debt-to-income ratio, on the same scale as the training data.")
    savings = c2.number_input("Savings", min_value=0.0, value=10000.0, step=500.0)
    collateral = c1.number_input("Collateral value", min_value=0.0, value=20000.0, step=500.0)

    st.subheader("Loan")
    c1, c2 = st.columns(2)
    purpose = c1.selectbox("Loan purpose", options["Loan_Purpose"])
    loan_amount = c2.number_input("Loan amount", min_value=1.0, value=15000.0, step=500.0)
    loan_term = c1.number_input("Loan term", min_value=1.0, value=36.0, step=1.0,
                                help="Same unit as the training data (e.g. months).")

    submitted = st.form_submit_button("Check approval odds", type="primary")

if submitted:
    payload = {
        "Applicant_Income": income,
        "Coapplicant_Income": co_income,
        "Age": int(age),
        "Dependents": int(dependents),
        "Credit_Score": float(credit_score),
        "Existing_Loans": int(existing_loans),
        "DTI_Ratio": dti,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Employment_Status": employment,
        "Marital_Status": marital,
        "Loan_Purpose": purpose,
        "Property_Area": area,
        "Gender": gender,
        "Employer_Category": employer,
        "Education_Level": education,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.exceptions.RequestException:
        st.error(f"Can't reach the API at {API_URL}.")
        st.stop()

    if resp.status_code != 200:
        st.error(f"The API rejected the request: {resp.json().get('detail', resp.text)}")
        st.stop()

    result = resp.json()
    probability = result["approval_probability"]
    if result["approved"]:
        st.success("Likely to be approved")
    else:
        st.error("Unlikely to be approved")
    st.metric("Approval probability", f"{probability:.1%}")
    st.progress(probability)
    st.caption("A model estimate for illustration, not a lending decision.")
