import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import re

st.set_page_config(page_title="Business Forecasting System", layout="wide")

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Sales Forecast", "Inventory Forecast", "Price Prediction"])


# ===============================
# FUNCTION: Train & Predict
# ===============================
def train_and_predict(df, column_name, future_days=50):
    df = df.copy()
    df["Days"] = np.arange(len(df))

    X = df[["Days"]]
    y = df[column_name]

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    predictions = model.predict(future_X)

    return df, predictions


# =========================================================
# PAGE 1: SALES FORECAST
# =========================================================
if page == "Sales Forecast":

    st.title("📊 Sales Forecast Prediction")

    df = pd.read_csv("sales_data_500.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    df, predictions = train_and_predict(df, "Sales")
    future_dates = pd.date_range(df["Date"].iloc[-1], periods=51, freq="D")[1:]

    chart_type = st.selectbox("Select Visualization Type", ["Line Chart", "Bar Chart"])

    fig, ax = plt.subplots()

    if chart_type == "Line Chart":
        ax.plot(df["Date"], df["Sales"], color="blue", label="Actual Sales")
        ax.plot(future_dates, predictions, color="yellow", label="Predicted Sales")
    else:
        ax.bar(df["Date"], df["Sales"], color="blue", label="Actual Sales")
        ax.bar(future_dates, predictions, color="yellow", label="Predicted Sales")

    ax.legend()
    ax.set_title("Sales Forecast (Next 50 Days)")
    st.pyplot(fig)

    # Chatbot
    st.subheader("🤖 Sales Chatbot")
    user_input = st.text_input("Ask: total sales next 20 days / average sales next 10 days")

    if user_input:
        user_input = user_input.lower()
        numbers = re.findall(r'\d+', user_input)

        if numbers:
            days = int(numbers[0])
            selected = predictions[:days]

            if "average" in user_input:
                result = np.mean(selected)
                st.success(f"Average predicted sales for next {days} days: {round(result,2)}")
            else:
                result = np.sum(selected)
                st.success(f"Total predicted sales for next {days} days: {round(result,2)}")
        else:
            st.warning("Please mention number of days.")


# =========================================================
# PAGE 2: INVENTORY FORECAST
# =========================================================
elif page == "Inventory Forecast":

    st.title("📦 Inventory Demand Forecast")

    df = pd.read_csv("inventory_data_500.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    df, predictions = train_and_predict(df, "Inventory_Required")
    future_dates = pd.date_range(df["Date"].iloc[-1], periods=51, freq="D")[1:]

    chart_type = st.selectbox("Select Visualization Type", ["Line Chart", "Bar Chart"])

    fig, ax = plt.subplots()

    if chart_type == "Line Chart":
        ax.plot(df["Date"], df["Inventory_Required"], color="blue", label="Actual Inventory")
        ax.plot(future_dates, predictions, color="yellow", label="Predicted Inventory")
    else:
        ax.bar(df["Date"], df["Inventory_Required"], color="blue", label="Actual Inventory")
        ax.bar(future_dates, predictions, color="yellow", label="Predicted Inventory")

    ax.legend()
    ax.set_title("Inventory Forecast (Next 50 Days)")
    st.pyplot(fig)

    # Chatbot
    st.subheader("🤖 Inventory Chatbot")
    user_input = st.text_input("Ask: inventory next 15 days / average inventory next 10 days")

    if user_input:
        user_input = user_input.lower()
        numbers = re.findall(r'\d+', user_input)

        if numbers:
            days = int(numbers[0])
            selected = predictions[:days]

            if "average" in user_input:
                result = np.mean(selected)
                st.success(f"Average inventory required for next {days} days: {round(result,2)}")
            else:
                result = np.sum(selected)
                st.success(f"Total inventory required for next {days} days: {round(result,2)}")
        else:
            st.warning("Please mention number of days.")


# =========================================================
# PAGE 3: PRICE PREDICTION
# =========================================================
elif page == "Price Prediction":

    st.title("💰 Product Price Prediction")

    df = pd.read_csv("price_data_500.csv")

    X = df[["Cost_Price", "Demand", "Rating"]]
    y = df["Selling_Price"]

    model = LinearRegression()
    model.fit(X, y)

    cost = st.number_input("Enter Cost Price", 100, 2000, 500)
    demand = st.number_input("Enter Demand", 50, 1000, 200)
    rating = st.slider("Enter Rating", 1.0, 5.0, 4.0)

    if st.button("Predict Price"):
        prediction = model.predict([[cost, demand, rating]])
        st.success(f"Predicted Selling Price: {round(prediction[0],2)}")

    # Graph
    fig, ax = plt.subplots()
    ax.plot(df["Selling_Price"].values, color="blue", label="Historical Prices")
    future_prices = model.predict(X.tail(50))
    ax.plot(range(len(df), len(df) + 50), future_prices, color="yellow", label="Predicted Trend")
    ax.legend()
    ax.set_title("Price Trend Prediction")
    st.pyplot(fig)

    # Chatbot
    st.subheader("🤖 Price Chatbot")
    user_input = st.text_input("Ask: price for demand 300")

    if user_input:
        user_input = user_input.lower()
        numbers = re.findall(r'\d+', user_input)

        if numbers:
            demand_val = int(numbers[0])
            pred = model.predict([[cost, demand_val, rating]])
            st.success(f"Predicted price for demand {demand_val}: {round(pred[0],2)}")
        else:
            st.warning("Please include demand value.")