# =========================
# SMARTOPS FORECAST - PHASE 1-2
# Multi-Product Forecasting using ARIMA and LSTM
# =========================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# ------------------------------
# 1️⃣ Load dataset
# ------------------------------
df = pd.read_csv("data/simulated_sales.csv", parse_dates=['Date'])
print("✅ Data Loaded Successfully")

# Ensure data is sorted
df = df.sort_values(by=['Product', 'Date'])

# Create folder for saving plots
os.makedirs("outputs/forecasts", exist_ok=True)

# Dictionary to store metrics
results = []

# ------------------------------
# 2️⃣ Loop through each product
# ------------------------------
for product in df['Product'].unique():
    print(f"\n📈 Processing Forecast for: {product}")
    
    product_df = df[df['Product'] == product].copy()
    series = product_df['Units_Sold'].values
    
    # Split data into train/test (80% train, 20% test)
    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]
    
    # ------------------------------
    # 3️⃣ ARIMA Forecast
    # ------------------------------
    arima_model = ARIMA(train, order=(2,1,2))  # order(p,d,q)
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=len(test))
    
    # Evaluation metrics
    arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
    arima_mape = mean_absolute_percentage_error(test, arima_forecast)
    print(f"ARIMA RMSE: {arima_rmse:.2f}, MAPE: {arima_mape:.2%}")
    
    # ------------------------------
    # 4️⃣ LSTM Forecast
    # ------------------------------
    n_input = 10  # time window for LSTM
    n_features = 1
    
    # Prepare generator for LSTM
    generator = TimeseriesGenerator(train, train, length=n_input, batch_size=8)
    
    # Define LSTM model
    lstm_model = Sequential([
        LSTM(64, activation='relu', input_shape=(n_input, n_features)),
        Dense(1)
    ])
    
    # Compile model
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # Train LSTM
    lstm_model.fit(generator, epochs=25, verbose=0)
    
    # Make predictions
    lstm_predictions = []
    test_seq = train[-n_input:]  # last window from train
    
    for i in range(len(test)):
        x_input = test_seq.reshape((1, n_input, n_features))
        yhat = lstm_model.predict(x_input, verbose=0)
        lstm_predictions.append(yhat[0][0])
        test_seq = np.append(test_seq[1:], yhat)  # slide window
    
    # Evaluation metrics
    lstm_rmse = np.sqrt(mean_squared_error(test, lstm_predictions))
    lstm_mape = mean_absolute_percentage_error(test, lstm_predictions)
    print(f"LSTM RMSE: {lstm_rmse:.2f}, MAPE: {lstm_mape:.2%}")
    
    # ------------------------------
    # 5️⃣ Save metrics & plot forecasts
    # ------------------------------
    results.append({
        "Product": product,
        "ARIMA_RMSE": arima_rmse,
        "ARIMA_MAPE": arima_mape,
        "LSTM_RMSE": lstm_rmse,
        "LSTM_MAPE": lstm_mape
    })
    
    plt.figure(figsize=(10,5))
    plt.plot(range(len(train)), train, label="Train")
    plt.plot(range(len(train), len(train)+len(test)), test, label="Actual")
    plt.plot(range(len(train), len(train)+len(test)), arima_forecast, label="ARIMA Forecast")
    plt.plot(range(len(train), len(train)+len(test)), lstm_predictions, label="LSTM Forecast")
    plt.title(f"{product} - Forecast Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/forecasts/{product}_forecast.png")
    plt.close()

# ------------------------------
# 6️⃣ Save all metrics to CSV
# ------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/forecast_results.csv", index=False)
print("\n✅ Forecasting Completed and Saved to outputs/forecast_results.csv")
