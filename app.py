"""
Streamlit app - Telco customer churn prediction.
Run:  streamlit run app.py   (after `python train.py`)
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/churn_model.joblib"
META_PATH = "models/metadata.json"

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


def risk_band(p: float) -> str:
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Medium"
    return "High"


def retention_tips(row: dict, p: float) -> list:
    """Simple rule-based retention suggestions driven by the top churn drivers."""
    tips = []
    if row["Contract"] == "Month-to-month":
        tips.append("Offer a discount to move from month-to-month to a 1- or 2-year contract.")
    if row["tenure"] <= 12:
        tips.append("New customer: run an onboarding / first-year loyalty programme.")
    if row["InternetService"] == "Fiber optic" and row["MonthlyCharges"] > 80:
        tips.append("High fiber bill: review the plan or bundle in a price lock.")
    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        tips.append("Offer free tech support / online security for a few months.")
    if row["PaymentMethod"] == "Electronic check":
        tips.append("Encourage auto-pay (card or bank transfer) with a small credit.")
    if not tips and p >= 0.30:
        tips.append("Proactive check-in call and a personalised loyalty offer.")
    if not tips:
        tips.append("No action needed - customer looks stable.")
    return tips


if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
    st.error("Model not found. Run `python train.py` first to train and save the model.")
    st.stop()

model, meta = load_artifacts()
threshold = meta["threshold"]
columns = meta["feature_columns"]

st.title("📉 Telecom Customer Churn Predictor")
st.caption(
    f"Model: **{meta['model_name']}**  |  Test ROC-AUC: **{meta['test_metrics']['roc_auc']:.3f}**  |  "
    f"Recall: **{meta['test_metrics']['recall']:.3f}**  |  Decision threshold: **{threshold:.2f}**"
)

tab_single, tab_batch, tab_insights = st.tabs(
    ["Single customer", "Batch scoring (CSV)", "Model insights"]
)

# ----------------------------- Single customer ----------------------------- #
with tab_single:
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.subheader("Services")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            no_net = "No internet service"
            sec = st.selectbox("Online security", ["No", "Yes", no_net])
            backup = st.selectbox("Online backup", ["No", "Yes", no_net])
            device = st.selectbox("Device protection", ["No", "Yes", no_net])
            tech = st.selectbox("Tech support", ["No", "Yes", no_net])
            tv = st.selectbox("Streaming TV", ["No", "Yes", no_net])
            movies = st.selectbox("Streaming movies", ["No", "Yes", no_net])

        with c3:
            st.subheader("Account")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=0.5)
            total = st.number_input(
                "Total charges ($)", 0.0, 10000.0, float(round(monthly * max(tenure, 1), 2)), step=1.0
            )

        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        row = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi,
            "InternetService": internet,
            "OnlineSecurity": sec,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        X_row = pd.DataFrame([row])[columns]
        proba = float(model.predict_proba(X_row)[0, 1])
        band = risk_band(proba)

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{proba:.1%}")
        m2.metric("Risk category", band)
        m3.metric("Predicted", "Will churn" if proba >= threshold else "Will stay")
        st.progress(min(proba, 1.0))

        st.subheader("Suggested retention actions")
        for tip in retention_tips(row, proba):
            st.write(f"- {tip}")

# ------------------------------ Batch scoring ------------------------------ #
with tab_batch:
    st.write(
        "Upload a CSV with the same columns as the Telco dataset "
        "(a `customerID` column is optional, `Churn` is ignored)."
    )
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        ids = data["customerID"] if "customerID" in data.columns else pd.Series(range(len(data)))
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0.0)
        try:
            X_batch = data[columns]
            probs = model.predict_proba(X_batch)[:, 1]
            out = pd.DataFrame(
                {
                    "customerID": ids,
                    "churn_probability": probs.round(4),
                    "risk": [risk_band(p) for p in probs],
                    "predicted_churn": (probs >= threshold).astype(int),
                }
            ).sort_values("churn_probability", ascending=False)

            st.success(f"Scored {len(out)} customers - {int(out['predicted_churn'].sum())} flagged as high churn risk.")
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download predictions", out.to_csv(index=False).encode(), "churn_predictions.csv"
            )
        except KeyError as e:
            st.error(f"Missing column in uploaded file: {e}")

# ------------------------------ Model insights ----------------------------- #
with tab_insights:
    st.subheader("Why do customers churn? (SHAP)")
    col_a, col_b = st.columns(2)
    if os.path.exists("models/shap_importance.png"):
        col_a.image("models/shap_importance.png", caption="Mean |SHAP| - global feature importance")
    if os.path.exists("models/shap_summary.png"):
        col_b.image("models/shap_summary.png", caption="SHAP summary - direction of impact")
    if os.path.exists("models/cv_results.csv"):
        st.subheader("Model comparison (5-fold stratified CV)")
        st.dataframe(
            pd.read_csv("models/cv_results.csv", index_col=0)[
                ["roc_auc", "pr_auc", "recall", "precision", "f1"]
            ].round(4),
            use_container_width=True,
        )
