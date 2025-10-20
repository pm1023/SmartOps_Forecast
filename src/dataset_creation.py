# =========================
# SMARTOPS FORECAST - PHASE 0
# Dataset Creation for Multi-Product Forecasting
# =========================

import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create folder for data if it doesn't exist
os.makedirs("data", exist_ok=True)

# Define products and date range
products = ['Product_A', 'Product_B', 'Product_C']
dates = pd.date_range(start="2024-01-01", periods=365)

# Create empty DataFrame to hold all products
all_data = pd.DataFrame()

# Generate simulated sales for each product
for product in products:
    # Base trend (increasing/decreasing)
    trend = np.linspace(20, 50, len(dates))
    
    # Seasonality: weekly pattern
    seasonality = 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
    
    # Random noise
    noise = np.random.normal(0, 5, len(dates))
    
    units_sold = trend + seasonality + noise
    
    product_df = pd.DataFrame({
        'Date': dates,
        'Product': product,
        'Units_Sold': np.round(units_sold).astype(int)
    })
    
    all_data = pd.concat([all_data, product_df])

# Save to CSV
all_data.to_csv("data/simulated_sales.csv", index=False)
print("✅ Simulated dataset created at data/simulated_sales.csv")
