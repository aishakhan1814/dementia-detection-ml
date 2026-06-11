
import streamlit as st
import joblib
import numpy as np

model = joblib.load("dementia_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Early Dementia Detection")
st.write("Enter patient details to predict dementia risk.")

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 18, 100, 70)
educ = st.selectbox("Education Level", [1, 2, 3, 4, 5], 
    format_func=lambda x: {1:"Less than High School", 2:"High School", 
    3:"Some College", 4:"College Graduate", 5:"Postgraduate"}[x])
ses = st.selectbox("Socioeconomic Status", [1, 2, 3, 4, 5],
    format_func=lambda x: {1:"Highest", 2:"Above Average", 
    3:"Average", 4:"Below Average", 5:"Lowest"}[x])
mmse = st.slider("MMSE Score (0-30)", 0, 30, 27)
etiv = st.number_input("eTIV (Intracranial Volume)", 1000, 2500, 1500)
nwbv = st.number_input("nWBV (Normalised Brain Volume)", 0.5, 1.0, 0.75)

if st.button("Predict"):
    gender_val = 1 if gender == "Male" else 0
    features = np.array([[gender_val, age, educ, ses, mmse, etiv, nwbv]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ High Dementia Risk Detected — Confidence: {probability:.1%}")
        st.write("Recommend immediate clinical evaluation.")
    else:
        st.success(f"✅ Low Dementia Risk — Confidence: {1-probability:.1%}")
        st.write("Continue routine monitoring.")
        
st.markdown("---")
st.caption("Based on OASIS-1 dataset. Not a substitute for clinical diagnosis.")
