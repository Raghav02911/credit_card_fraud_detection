import streamlit as st
import joblib
import numpy as np

#load the model
model = joblib.load('model/random_forest_model.pkl')

st.title("Credit Card Fraud Detection App")
st.write("Enter the transaction details to check for fraud.")

#sample data
legitimate_sample = {
    'time': 100000.0,
    'amount': 50.0,
    'v': [-1.2, 0.5, 1.3, -0.8, 0.2, -0.5, 0.9, 0.1, -0.3, 0.7, -1.0, 0.4, 0.6, -0.2, 0.8, -0.6, 0.3, 0.9, -0.4, 0.1, 0.5, -0.7, 0.2, 0.8, -0.9, 0.6, -0.1, 0.4]
}

fraud_sample = {
    'time': 406.0,
    'amount': 0.0,
    'v': [-2.3122265423263, 1.95199201064158, -1.60985073229769, 3.9979055875468, -0.522187864667764, -1.42654531920595, -2.53738730624579, 1.39165724829804, -2.77008927719433, -2.77227214465915, 3.20203320709635, -2.89990738849473, -0.595221881324605, -4.28925378244217, 0.389724120274487, -1.14074717980657, -2.83005567450437, -0.0168224681808257, 0.416955705037907, 0.126910559061474, 0.517232370861764, -0.0350493686052974, -0.465211076182388, 0.320198198514526, 0.0445191674731724, 0.177839798284401, 0.261145002567677, -0.143275874698919]
}

#input methods
option = st.selectbox("Choose input method:", ["Custom Input", "Sample Legitimate Transaction", "Sample Fraudulent Transaction"])

if option == "Custom Input":
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

    input_data = np.array([[time] + features + [amount]])

elif option == "Sample Legitimate Transaction":
    st.write("**Sample Legitimate Transaction Values:**")
    st.write(f"Time: {legitimate_sample['time']}")
    st.write(f"Amount: {legitimate_sample['amount']}")
    st.write("V values:", legitimate_sample['v'])
    input_data = np.array([[legitimate_sample['time']] + legitimate_sample['v'] + [legitimate_sample['amount']]])

elif option == "Sample Fraudulent Transaction":
    st.write("**Sample Fraudulent Transaction Values:**")
    st.write(f"Time: {fraud_sample['time']}")
    st.write(f"Amount: {fraud_sample['amount']}")
    st.write("V values:", fraud_sample['v'])
    input_data = np.array([[fraud_sample['time']] + fraud_sample['v'] + [fraud_sample['amount']]])

#predict button
if st.button("Predict Fraud"):
    #prediction
    prob_fraud = model.predict_proba(input_data)[0][1]
    prediction = 1 if prob_fraud > 0.45 else 0
    
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! Probability: {prob_fraud:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Probability: {prob_fraud:.2f}")