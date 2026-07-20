import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("GradientBoosting_model.pkl")

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Stock Price Prediction")
st.write("Enter the stock details below to predict the closing price.")

# User Inputs
open_price = st.number_input("Open Price", min_value=0.0, format="%.2f")
high = st.number_input("High Price", min_value=0.0, format="%.2f")
low = st.number_input("Low Price", min_value=0.0, format="%.2f")
prev_close = st.number_input("Previous Close", min_value=0.0, format="%.2f")
tottrdqty = st.number_input("Total Traded Quantity", min_value=0.0, format="%.2f")
tottrdval = st.number_input("Total Traded Value", min_value=0.0, format="%.2f")
total_trades = st.number_input("Total Trades", min_value=0.0, format="%.2f")

# Prediction Button
if st.button("Predict Closing Price"):

    data = pd.DataFrame([[
        open_price,
        high,
        low,
        prev_close,
        tottrdqty,
        tottrdval,
        total_trades
    ]], columns=[
        "OPEN",
        "HIGH",
        "LOW",
        "PREV. CLOSE",
        "Tottrdqty",
        "Tottrdval",
        "Total_Trades"
    ])

    prediction = model.predict(data)

    st.success(f"Predicted Closing Price: ₹ {prediction[0]:.2f}")

    st.subheader("Input Data")
    st.dataframe(data)