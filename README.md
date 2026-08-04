# ⚡ Energy Consumption Forecasting with PyTorch LSTM

A modular, production-ready Deep Learning pipeline to predict hourly energy consumption (AEP energy dataset) using a Multi-Layer Long Short-Term Memory (LSTM) network built with PyTorch.

---

## 📌 Project Overview

Accurate short-term load forecasting is essential for power grid stability and energy management. This project builds an end-to-end time-series forecasting workflow:
- **Data Engineering**: Processes raw hourly energy consumption data, creates lag features/rolling statistics, and handles train/validation/test splits.
- **Deep Learning Model**: Implements a configurable stacked LSTM architecture (`EnergyLSTM`) trained using PyTorch.
- **Inference Ready**: Includes a robust standalone prediction class for unscaling model outputs into real-world physical units (e.g., MW/kW).

---

## 📂 Project Structure

```text
Energy_Consumption/
├── data/
│   ├── raw/               # Raw AEP hourly dataset
│   └── processed/         # Feature-engineered dataset
├── models/                # Saved weights (.pth) & Scikit-Learn scalers (.pkl)
├── notebooks/
│   ├── eda_and_cleaning.ipynb # EDA, visualization, and cleaning
│   └── evaluation.ipynb       # Model evaluation & actual vs. predicted plots
├── reports/
│   └── figures/           # Generated metric plots and charts
├── src/
│   ├── models/
│   │   └── lstm_model.py  # PyTorch LSTM network architecture
│   ├── dataset.py         # Sliding window Dataset & DataLoaders
│   ├── features.py        # Scalers, splits, and feature generation
│   ├── train.py           # Training loop with validation checkpointing
│   └── predict.py         # End-to-end inference class for raw input data
├── .gitignore
├── pyproject.toml         # UV project dependencies configuration
├── README.md              # Project documentation
└── uv.lock                # Locked dependency graph