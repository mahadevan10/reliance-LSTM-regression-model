
# Reliance LSTM Regression Model

Predicting Stock Price Movements with Deep Learning

## Overview
This project leverages advanced deep learning techniques to predict whether Reliance stock will close higher or lower the next day. It demonstrates a full machine learning workflow, from data preprocessing and feature engineering to model training, evaluation, and visualization.

## Key Features & Techniques
- **Data Preprocessing:**
  - Handling missing values, scaling features (MinMaxScaler), and transforming time series data for supervised learning.
- **Feature Engineering:**
  - Creation of technical indicators (moving averages, returns, volatility, RSI, MACD).
  - Lagged features and rolling statistics to capture temporal dependencies.
- **Exploratory Data Analysis (EDA):**
  - Visualizations: Volume vs Price, Moving Average relationships, 3D scatter plots, hexbin plots, boxplots, and more.
  - Statistical summaries and correlation analysis.
- **Modeling:**
  - Implementation of Long Short-Term Memory (LSTM) neural networks for sequence prediction.
  - Comparison with baseline models (Random Forest, Logistic Regression).
- **Evaluation:**
  - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
  - Visualization of predictions and error analysis.
- **Deployment Ready:**
  - Modular code structure for easy extension and integration.
  - Jupyter notebooks for reproducibility and interactive analysis.

## Project Structure
- `src/`: Source code modules
  - `data_preprocessing.py`: Data cleaning and transformation
  - `feature_engineering.py`: Feature creation and selection
  - `lstm_model.py`: LSTM model definition and training
  - `model_evaluation.py`: Model evaluation and metrics
  - `utils.py`: Helper functions
- `data/`:
  - `raw/`: Raw data files
  - `processed/`: Cleaned and transformed data
- `models/`: Saved model files (.h5, .pkl)
- `images/`: Plots and figures generated during analysis
- `notebooks/`: Jupyter notebooks for EDA and experiments
- `requirements.txt`: Python dependencies

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your data in the `data/raw/` folder
4. Run the notebooks or scripts in `src/` to preprocess data, engineer features, and train models

## Results
- The accuracy is low but in the real market scenario, financial markets are completely random in day to day movements but i was able to bring r2 score to a minimum after experimentaions with the model tuning and feature engineering
- Visualizations reveal key relationships between volume, price, and technical indicators
- Modular code allows for easy experimentation with new features and models




