import streamlit as st
import pandas as pd
import pickle
import yfinance as yf
# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Stock Volume Prediction",
    page_icon="📈",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    with open("GradientBoosting_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------- TITLE --------------------
st.title("📈 Stock Volume Prediction")
symbol = st.text_input(
    "Enter Stock Symbol",
    value="TCS.NS"
)
if symbol:

    stock = yf.Ticker(symbol)

    data = stock.history(period="2d")

    if not data.empty:

        latest = data.iloc[-1]

        open_price = latest["Open"]
        high = latest["High"]
        low = latest["Low"]
        close = latest["Close"]
        volume = latest["Volume"]
        st.subheader("Live Stock Data")

st.write("Open:", round(open_price, 2))
st.write("High:", round(high, 2))
st.write("Low:", round(low, 2))
st.write("Current Price:", round(close, 2))
st.write("Volume:", int(volume))
        input_data = pd.DataFrame(
            [[
                open_price,
                high,
                low,
                close,
                volume
            ]],
            columns=[
                "OPEN",
                "HIGH",
                "LOW",
                "PREV. CLOSE",
                "VOLUME (shares)"
            ]
        )

        prediction = model.predict(input_data)

        st.success(f"Predicted Price: ₹{prediction[0]:.2f}")
st.write("Enter the stock details below.")

# ======================================================
# IMPORTANT
# Replace this dictionary with YOUR LabelEncoder mapping
# ======================================================
symbol_to_int = {
    "RELIANCE": 0,
    "TCS": 1,
    "INFY": 2,
    "HDFCBANK": 3,
    "ICICIBANK": 4,
}

# -------------------- SIDEBAR --------------------
st.sidebar.header("Input Features")

symbol = st.sidebar.selectbox(
    "SYMBOL",
    list(symbol_to_int.keys())
)

open_price = st.sidebar.number_input("OPEN", value=1000.0)
high = st.sidebar.number_input("HIGH", value=1010.0)
low = st.sidebar.number_input("LOW", value=990.0)
prev_close = st.sidebar.number_input("PREV. CLOSE", value=995.0)
ltp = st.sidebar.number_input("LTP", value=1002.0)
indicative_close = st.sidebar.number_input("INDICATIVE CLOSE", value=1002.0)
change = st.sidebar.number_input("CHANGE", value=5.0)
percent_change = st.sidebar.number_input("% CHANGE", value=0.50)
value = st.sidebar.number_input("VALUE (Crores)", value=100.0)
high52 = st.sidebar.number_input("52W H", value=1200.0)
low52 = st.sidebar.number_input("52W L", value=800.0)
change30 = st.sidebar.number_input("30 D %CHNG", value=2.5)
change365 = st.sidebar.number_input("365 D %CHNG", value=12.0)

symbol_encoded = symbol_to_int.get(symbol, 0)

features = pd.DataFrame({
    "SYMBOL": [symbol_encoded],
    "OPEN": [open_price],
    "HIGH": [high],
    "LOW": [low],
    "PREV. CLOSE": [prev_close],
    "LTP": [ltp],
    "INDICATIVE CLOSE": [indicative_close],
    "CHANGE": [change],
    "% CHANGE": [percent_change],
    "VALUE (Crores)": [value],
    "52W H": [high52],
    "52W L": [low52],
    "30 D %CHNG": [change30],
    "365 D %CHNG": [change365],
})

features = features.reindex(columns=model.feature_names_in_)

st.subheader("Input Data")
st.dataframe(features)

if st.button("Predict"):

    try:
        prediction = model.predict(features)

        st.success("Prediction Successful!")

        st.metric(
            label="Predicted Volume",
            value=f"{prediction[0]:,.0f}"
        )

    except Exception as e:
        st.error(e)

        st.write("Model expects these columns:")
        st.write(model.feature_names_in_)

        st.write("Columns received:")
        st.write(features.columns.tolist())

        st.write("Data Types:")
        st.write(features.dtypes)
