import streamlit as st
import pandas as pd
import pickle

# Load scaler and model
scaler = pickle.load(open("models/scaler.pkl", "rb"))
model = pickle.load(open("models/model_gbc.pkl", "rb"))

st.title("Chronic Disease Prediction")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120)
bp = st.number_input("Blood Pressure", min_value=0)
sg = st.number_input("Specific Gravity", value=1.015, format="%.3f")
al = st.number_input("Albumin Level", min_value=0.0, step=0.1)
hemo = st.number_input("Hemoglobin Level", min_value=0.0, step=0.1)
sc = st.number_input("Serum Creatinine", min_value=0.0, step=0.01)

htn = st.selectbox("Hypertension", ["yes", "no"])
dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
appet = st.selectbox("Appetite", ["good", "poor"])
pc = st.selectbox("Pus Cell", ["normal", "abnormal"])

# Prediction logic
if st.button("Predict"):
    # Prepare data
    data = pd.DataFrame({
        'age': [age],
        'bp': [bp],
        'sg': [sg],
        'al': [al],
        'hemo': [hemo],
        'sc': [sc],
        'htn': [1 if htn == "yes" else 0],
        'dm': [1 if dm == "yes" else 0],
        'cad': [1 if cad == "yes" else 0],
        'appet': [1 if appet == "good" else 0],
        'pc': [1 if pc == "normal" else 0]
    })

    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    prediction = model.predict(data)[0]
    
    st.success("Prediction: Chronic Disease Detected" if prediction == 1 else "Prediction: No Chronic Disease")
