# SmartOps Forecast

## Overview
SmartOps Forecast predicts product demand using ARIMA and LSTM, and optimizes multi-product inventory using Reinforcement Learning. It includes an interactive Streamlit dashboard for scenario exploration.

## Features
- Multi-product demand forecasting
- ARIMA and LSTM models for comparison
- Multi-product DQN agent for inventory optimization
- Interactive Streamlit dashboard
- SQL/database ready (optional for real-world integration)

## Folder Structure
- `data/` : sales dataset
- `src/` : scripts for dataset, forecasting, RL environment, and agent
- `outputs/` : forecast and RL plots
- `dashboard/` : Streamlit dashboard
- `requirements.txt` : all required Python packages

## Usage
1. Install packages: `pip install -r requirements.txt`
2. Generate dataset: `python src/dataset_creation.py`
3. Run forecasting: `python src/forecasting.py`
4. Train RL agent: `python src/train_rl.py`
5. Launch dashboard: `streamlit run dashboard/app.py`
