# =========================
# SMARTOPS FORECAST - PHASE 5
# Interactive Dashboard using Streamlit
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣ Load forecast results and sales data
# ------------------------------
sales_df = pd.read_csv("data/simulated_sales.csv", parse_dates=['Date'])
forecast_df = pd.read_csv("outputs/forecast_results.csv")

products = sales_df['Product'].unique()

st.title("🛒 SmartOps Forecast Dashboard")
st.markdown("Forecast vs Actual Sales & RL Inventory Decisions")

# ------------------------------
# 2️⃣ Sidebar options
# ------------------------------
selected_product = st.sidebar.selectbox("Select Product", products)
safety_stock = st.sidebar.slider("Safety Stock", min_value=0, max_value=20, value=5)
order_frequency = st.sidebar.slider("Order Frequency (days)", min_value=1, max_value=7, value=3)

# ------------------------------
# 3️⃣ Filter product data
# ------------------------------
product_sales = sales_df[sales_df['Product'] == selected_product].sort_values('Date')
arima_forecast = forecast_df[forecast_df['Product'] == selected_product]['ARIMA_RMSE'].values[0]
lstm_forecast = forecast_df[forecast_df['Product'] == selected_product]['LSTM_RMSE'].values[0]

# ------------------------------
# 4️⃣ Plot forecast vs actual sales
# ------------------------------
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(product_sales['Date'], product_sales['Units_Sold'], label="Actual Sales")
ax.set_title(f"{selected_product} - Sales with Safety Stock {safety_stock}")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
st.pyplot(fig)

# ------------------------------
# 5️⃣ Simulate simple inventory policy
# ------------------------------
inventory = []
stock = safety_stock  # initial inventory

for units in product_sales['Units_Sold']:
    if stock < units:
        order_qty = max(0, units - stock)
        stock += order_qty
    stock -= units
    inventory.append(stock)

# ------------------------------
# 6️⃣ Plot inventory over time
# ------------------------------
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(product_sales['Date'], inventory, label="Inventory Level", color='orange')
ax2.axhline(safety_stock, color='red', linestyle='--', label="Safety Stock")
ax2.set_title(f"{selected_product} - Inventory Simulation")
ax2.set_xlabel("Date")
ax2.set_ylabel("Inventory Units")
ax2.legend()
st.pyplot(fig2)

# ------------------------------
# 7️⃣ Display metrics
# ------------------------------
st.subheader("Forecast Metrics")
st.markdown(f"

