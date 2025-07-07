
import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open('credit_model.pkl', 'rb'))

st.title("Credit Risk Predictor")

# Example inputs (you will replace these with real ones)
Age = st.number_input("Enter Age")
Income = st.number_input("Enter Monthly Income")
Credit_Score = st.slider("Credit Score", 300, 850)

# You can add more fields depending on your model's input

# Prepare input
input_df = pd.DataFrame([[Age, Income, Credit_Score]],
                        columns=["Age", "Income", "Credit_Score"])

# Predict
if st.button("Predict"):
    result = model.predict(input_df)
    st.success(f"Predicted credit risk: {result[0]}")
