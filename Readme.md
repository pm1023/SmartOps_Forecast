# 🚀 SmartOps Forecast

[![Repo](https://img.shields.io/badge/repo-SmartOps__Forecast-blue)](https://github.com/pm1023/SmartOps_Forecast)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

SmartOps Forecast predicts multi-product demand (ARIMA & LSTM) and optimizes multi-product inventory using Reinforcement Learning (DQN). It includes an interactive Streamlit dashboard for scenario exploration and is ready to integrate with SQL/databases for production use.

---

## ✨ Key Features
- Accurate multi-product demand forecasting
  - Compare ARIMA and LSTM models
- Inventory optimization using a Multi-Product DQN agent
- Interactive Streamlit dashboard for scenario analysis
- Modular codebase: easy to extend, plug into databases, or swap models
- Jupyter/CLI-friendly examples and visual outputs in `outputs/`

---

## 📁 Project Structure
- data/ — sales dataset and sample inputs  
- src/ — scripts for dataset generation, forecasting, RL environment, and agents  
- outputs/ — generated plots, model artifacts and visualizations  
- dashboard/ — Streamlit app (dashboard/app.py)  
- requirements.txt — Python dependencies

---

## 🛠️ Quick Start

1. Clone the repository
```bash
git clone https://github.com/pm1023/SmartOps_Forecast.git
cd SmartOps_Forecast
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Generate a sample dataset
```bash
python src/dataset_creation.py
```

4. Run forecasting (ARIMA / LSTM)
```bash
python src/forecasting.py
# outputs/ will contain forecast plots and metrics
```

5. Train the RL agent for inventory optimization
```bash
python src/train_rl.py
# Trained agent + training curves saved to outputs/
```

6. Launch the interactive dashboard
```bash
streamlit run dashboard/app.py
# Open http://localhost:8501 to explore scenarios and visualize results
```

---

## 🔍 What you’ll see
- Forecast plots and performance metrics for individual products
- Comparison view: ARIMA vs LSTM forecasts
- RL training curves and policy evaluation
- Dashboard controls for lead time, service level, holding/stockout costs, and demand scenarios

---

## 🧭 Recommended Workflow
1. Create or load historical sales in `data/`
2. Run dataset preprocessing: `src/dataset_creation.py`
3. Train/evaluate forecasting models: `src/forecasting.py`
4. Simulate/optimize inventory with RL: `src/train_rl.py`
5. Visualize and explore scenarios with Streamlit: `dashboard/app.py`

---

## 🧪 Tips & Notes
- Models are configurable in `src/` — adjust hyperparameters for LSTM and ARIMA there.
- For production, swap the sample data loader with a database connector (SQL support is ready).
- Save best models and checkpoints to `outputs/models/` for reproducibility.

---

## 🧩 Extending the Project
- Add seasonal models or Prophet for improved forecasting
- Implement multi-agent RL for decentralized inventory systems
- Integrate with a CI pipeline and model registry for automated retraining

---

## 🤝 Contributing
Contributions welcome! Please open issues for bugs or feature requests, and submit PRs for fixes or enhancements.

- Create an issue describing what you want to change
- Fork the repo, make your changes on a branch, and open a PR referencing the issue

---

## 📄 License
MIT License — see LICENSE file.

---

## 📬 Contact
Built by pm1023. For questions or help, open an issue or contact via GitHub profile.

Happy forecasting! 📈
