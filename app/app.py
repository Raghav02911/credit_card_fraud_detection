import streamlit as st
import joblib
import numpy as np

#load the model
model = joblib.load('../model/random_forest_model.pkl')

st.title("Credit Card Fraud Detection App")
st.write("Enter the transaction details to check for fraud.")

#input fields
time = st.number_input("Time", value=0.0, step=0.01)
amount = st.number_input("Amount", value=0.0, step=0.01)

st.write("Enter V1 to V28 values:")
cols = st.columns(4)
features = []
for i in range(1, 29):
    with cols[(i-1) % 4]:
        val = st.number_input(f"V{i}", value=0.0, step=0.01, key=f"v{i}")
        features.append(val)

#button to predict
if st.button("Predict Fraud"):
    # Prepare the input array
    input_data = np.array([[time] + features + [amount]])
    
    #make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Probability: {probability:.2f}")