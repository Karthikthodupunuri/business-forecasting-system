import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Business Intelligence System", layout="wide")

st.title("📊 AI Business Intelligence Dashboard")

# ==============================
# 📂 DATA UPLOAD SYSTEM
# ==============================

st.sidebar.header("📂 Upload Dataset")

uploaded_sales = st.sidebar.file_uploader("Upload Sales CSV", type=["csv"])
uploaded_price = st.sidebar.file_uploader("Upload Price CSV", type=["csv"])

if uploaded_sales:
    sales_df = pd.read_csv(uploaded_sales)
else:
    sales_df = pd.read_csv("sales_data.csv")

if uploaded_price:
    price_df = pd.read_csv(uploaded_price)
else:
    price_df = pd.read_csv("price_data.csv")

# ==============================
# 🧭 NAVIGATION
# ==============================

page = st.sidebar.radio(
    "Navigate",
    ["Sales Forecast", "Inventory Management", 
     "Price Prediction", "Business Chatbot"]
)

# ==============================
# 📈 SALES FORECAST (LINEAR)
# ==============================

if page == "Sales Forecast":

    st.header("📈 Sales Forecast (Linear Regression)")

    sales_df["Date"] = pd.to_datetime(sales_df["Date"])
    sales_df = sales_df.sort_values("Date")

    sales_df["Day"] = np.arange(len(sales_df))

    X = sales_df[["Day"]]
    y = sales_df["Sales"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("R² Score", round(r2,3))
    col2.metric("MAE", round(mae,2))

    fig = px.line(sales_df, x="Date", y="Sales",
                  title="Historical Sales")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Future Sales")

    forecast_days = st.slider("Select Forecast Days", 10, 120, 30)

    future_days = np.arange(len(sales_df), len(sales_df)+forecast_days)
    forecast = model.predict(future_days.reshape(-1,1))

    future_dates = pd.date_range(
        sales_df["Date"].iloc[-1],
        periods=forecast_days+1,
        freq="D"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast
    })

    fig2 = px.line(forecast_df, x="Date", y="Forecast",
                   title="Future Sales Forecast")
    st.plotly_chart(fig2, use_container_width=True)

# ==============================
# 📦 INVENTORY PAGE
# ==============================

elif page == "Inventory Management":

    st.header("📦 Inventory Management")

    total_sales = sales_df["Sales"].sum()
    avg_sales = sales_df["Sales"].mean()

    col1, col2 = st.columns(2)
    col1.metric("Total Sales", round(total_sales,2))
    col2.metric("Average Sales", round(avg_sales,2))

    st.subheader("Stock Status Simulation")

    stock = st.number_input("Current Stock", min_value=0)

    if stock < avg_sales:
        st.error("⚠ Low Stock Warning!")
    else:
        st.success("Stock Level is Healthy.")

# ==============================
# 💰 PRICE PREDICTION
# ==============================

elif page == "Price Prediction":

    st.header("💰 Smart Price Prediction")

    X = price_df[["Cost", "Demand", "Competitor_Price"]]
    y = price_df["Price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_option = st.selectbox(
        "Choose Model",
        ["Linear Regression", "Random Forest"]
    )

    if model_option == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100)

    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("R² Score", round(r2,3))
    col2.metric("MAE", round(mae,2))

    st.subheader("Enter Product Details")

    cost = st.number_input("Cost", min_value=0.0)
    demand = st.number_input("Demand", min_value=0.0)
    competitor = st.number_input("Competitor Price", min_value=0.0)

    if st.button("Predict Price"):
        input_data = scaler.transform([[cost, demand, competitor]])
        prediction = model.predict(input_data)
        st.success(f"Recommended Price: ₹ {round(prediction[0],2)}")

# ==============================
# 🤖 BUSINESS CHATBOT
# ==============================

elif page == "Business Chatbot":

    st.header("🤖 AI Business Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask about sales, forecast, demand, price...")

    if user_input:

        user_input = user_input.lower()

        if "total sales" in user_input:
            response = f"Total Sales: {round(sales_df['Sales'].sum(),2)}"

        elif "average sales" in user_input:
            response = f"Average Sales: {round(sales_df['Sales'].mean(),2)}"

        elif "forecast" in user_input:
            future_days = np.arange(len(sales_df), len(sales_df)+30)
            model = LinearRegression()
            sales_df["Day"] = np.arange(len(sales_df))
            model.fit(sales_df[["Day"]], sales_df["Sales"])
            future = model.predict(future_days.reshape(-1,1))
            response = f"Next 30 Days Forecast Total: {round(sum(future),2)}"

        elif "best price" in user_input:
            response = f"Average Market Price: ₹ {round(price_df['Price'].mean(),2)}"

        elif "high demand" in user_input:
            max_demand = price_df.loc[price_df["Demand"].idxmax()]
            response = f"Highest Demand Product Price: ₹ {max_demand['Price']}"

        else:
            response = "I can help with sales, forecast, pricing and demand analysis."

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.write(f"🧑 {msg}")
        else:
            st.write(f"🤖 {msg}")
