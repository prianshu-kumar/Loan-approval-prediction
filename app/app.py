import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"

@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
)

st.markdown("""
# Loan Approval Predictor
Use the form below to simulate an application decision based on your saved model.
""")

st.markdown("---")

with st.sidebar:
    st.header("About this app")
    st.write(
        "This Streamlit app loads `model/model.pkl` and predicts loan approval using the same features from your notebook."
    )
    st.write("---")
    st.markdown("**How to use**")
    st.write(
        "1. Enter applicant details.\n"
        "2. Click **Predict**.\n"
        "3. The app shows approval status and probability if available."
    )
    st.write("---")
    st.write("**Model file**")
    st.code(str(MODEL_PATH), language="bash")

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Place model.pkl in the model folder.")
    st.stop()
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

with st.expander("Example input reference from dataset", expanded=False):
    st.write(
        {
            "Gender": "Male",
            "Married": "No",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5849,
            "CoapplicantIncome": 0.0,
            "LoanAmount": 128.0,
            "Loan_Amount_Term": 360.0,
            "Credit_History": 1.0,
            "Property_Area": "Urban",
        }
    )

form_cols = st.columns([1, 1, 1])
with st.form("loan_form"):
    with form_cols[0]:
        loan_id = st.text_input("Loan ID", value="LP000001")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

    with form_cols[1]:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
        credit_history = st.selectbox("Credit History", ["1.0", "0.0"])

    with form_cols[2]:
        applicant_income = st.number_input(
            "Applicant Income", min_value=0, value=5000, step=100, format="%d"
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income", min_value=0.0, value=0.0, step=50.0, format="%.2f"
        )
        loan_amount = st.number_input(
            "Loan Amount", min_value=0.0, value=120.0, step=10.0, format="%.2f"
        )
        loan_amount_term = st.number_input(
            "Loan Amount Term", min_value=0.0, value=360.0, step=12.0, format="%.0f"
        )

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Loan_ID": loan_id,
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": float(credit_history),
                "Property_Area": property_area,
            }
        ]
    )

    try:
        prediction = model.predict(input_data)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)
            score = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]
        else:
            score = None

        approved = prediction[0] in ["Y", 1, "1"]
        label = "Approved" if approved else "Denied"
        status_text = "✅ Loan would likely be approved" if approved else "❌ Loan would likely be denied"

        if approved:
            st.success(status_text)
        else:
            st.error(status_text)

        if score is not None:
            st.metric("Approval probability", f"{score:.1%}")

        # with st.container():
        #     st.markdown("### Input summary")
        #     st.write(input_data.T.rename(columns={0: "Value"}))

        with st.expander("Model output details"):
            st.write({"prediction": prediction[0], "confidence": score})

