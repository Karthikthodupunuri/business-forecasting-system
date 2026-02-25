import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import re

st.set_page_config(page_title="Business Forecasting System", layout="wide")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Sales Forecast", "Inventory Forecast", "Price Prediction"])
theme = st.sidebar.radio("Theme", ["Dark", "Light"])

template_style = "plotly_dark" if theme == "Dark" else "plotly_white"


# ===============================
# TRAIN FUNCTION
# ===============================
def train_model(df, column_name):
    df["Days"] = np.arange(len(df))
    X = df[["Days"]]
    y = df[column_name]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return model, r2, mae


# ===============================
# COMMON FORECAST FUNCTION
# ===============================
def forecast(model, df, future_days):
    future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    predictions = model.predict(future_X)
    return predictions


# =========================================================
# SALES PAGE
# =========================================================
if page == "Sales Forecast":

    st.title("📊 Sales Forecast Dashboard")

    df = pd.read_csv("sales_data_500.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    future_days = st.slider("Select Forecast Days", 10, 120, 50)

    model, r2, mae = train_model(df, "Sales")
    predictions = forecast(model, df, future_days)

    future_dates = pd.date_range(df["Date"].iloc[-1], periods=future_days+1, freq="D")[1:]

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Last Actual Sales", round(df["Sales"].iloc[-1], 2))
    col2.metric("Next Day Prediction", round(predictions[0], 2))
    col3.metric("Model R² Score", round(r2, 3))

    # Graph Type
    chart_type = st.selectbox(
        "Visualization Type",
        ["Line Chart", "Scatter Plot", "Area Chart"]
    )

    fig = go.Figure()

    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Sales"],
                                 mode='lines', name="Actual",
                                 line=dict(color='blue', width=3)))
    elif chart_type == "Scatter Plot":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Sales"],
                                 mode='markers', name="Actual",
                                 marker=dict(color='blue')))
    elif chart_type == "Area Chart":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Sales"],
                                 fill='tozeroy', mode='lines',
                                 name="Actual",
                                 line=dict(color='blue')))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name="Predicted",
        line=dict(color='yellow', width=3, dash='dash')
    ))

    fig.update_layout(
        title="Sales Forecast",
        template=template_style
    )

    st.plotly_chart(fig, use_container_width=True)

    # Accuracy
    st.subheader("📈 Model Performance")
    st.write(f"R² Score: {round(r2,3)}")
    st.write(f"MAE: {round(mae,2)}")

    # Business Insight
    st.subheader("📊 Business Insight")
    increase = predictions[0] - df["Sales"].iloc[-1]
    if increase > 0:
        st.success(f"Sales expected to increase by {round(increase,2)} units.")
    else:
        st.warning(f"Sales expected to decrease by {round(abs(increase),2)} units.")

    # Download
    pred_df = pd.DataFrame({
        "Future_Date": future_dates,
        "Predicted_Sales": predictions
    })

    st.download_button("Download Forecast CSV",
                       pred_df.to_csv(index=False),
                       "sales_forecast.csv",
                       "text/csv")

    # Chatbot
    st.subheader("🤖 Smart Assistant")
    user_input = st.text_input("Ask: total sales next 20 days / average next 10 days")

    if user_input:
        user_input = user_input.lower()
        numbers = re.findall(r'\d+', user_input)

        if numbers:
            days = int(numbers[0])
            selected = predictions[:days]

            if "average" in user_input:
                st.success(f"Average: {round(np.mean(selected),2)}")
            elif "max" in user_input:
                st.success(f"Maximum: {round(np.max(selected),2)}")
            elif "min" in user_input:
                st.success(f"Minimum: {round(np.min(selected),2)}")
            else:
                st.success(f"Total: {round(np.sum(selected),2)}")
        else:
            st.warning("Please include number of days.")


# =========================================================
# INVENTORY PAGE (Same logic applied)
# =========================================================
elif page == "Inventory Forecast":

    st.title("📦 Inventory Forecast Dashboard")

    df = pd.read_csv("inventory_data_500.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    future_days = st.slider("Select Forecast Days", 10, 120, 50)

    model, r2, mae = train_model(df, "Inventory_Required")
    predictions = forecast(model, df, future_days)

    future_dates = pd.date_range(df["Date"].iloc[-1], periods=future_days+1, freq="D")[1:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["Date"], y=df["Inventory_Required"],
                             mode='lines',
                             name="Actual",
                             line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=future_dates, y=predictions,
                             mode='lines',
                             name="Predicted",
                             line=dict(color='yellow', dash='dash')))

    fig.update_layout(template=template_style)

    st.plotly_chart(fig, use_container_width=True)

    st.write(f"R² Score: {round(r2,3)} | MAE: {round(mae,2)}")


# =========================================================
# PRICE PAGE
# =========================================================
elif page == "Price Prediction":

    st.title("💰 Price Prediction Dashboard")

    df = pd.read_csv("price_data_500.csv")

    X = df[["Cost_Price", "Demand", "Rating"]]
    y = df["Selling_Price"]

    model = LinearRegression()
    model.fit(X, y)

    cost = st.number_input("Cost Price", 100, 2000, 500)
    demand = st.number_input("Demand", 50, 1000, 200)
    rating = st.slider("Rating", 1.0, 5.0, 4.0)

    if st.button("Predict"):
        prediction = model.predict([[cost, demand, rating]])
        st.success(f"Predicted Selling Price: {round(prediction[0],2)}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=df["Selling_Price"],
        mode='lines',
        name="Historical",
        line=dict(color='blue')
    ))

    fig.update_layout(template=template_style)

    st.plotly_chart(fig, use_container_width=True)
