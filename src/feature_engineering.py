from data_preprocessing import preprocess_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas_ta as ta
import joblib
import os
import numpy as np


# Preprocess data
df = preprocess_data()  # Index = Date, must include 'Close'

# Ensure data is sorted by date (index)
df = df.sort_index()

# Create target: closing price 2 days after today
df['Target'] = df['Close'].shift(-2)
df.dropna(inplace=True)

# Scale the 'Close' feature
scaler = StandardScaler()
X = scaler.fit_transform(df[['Close']].values)
y = df['Target'].values  # Unscaled

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], 1, 1))

# Save X, y, and scaler
save_path = r"E:\randomForestClassifier\reliance-LSTM-classifier\data"
os.makedirs(save_path, exist_ok=True)
joblib.dump(X, os.path.join(save_path, "X_lstm.pkl"))
joblib.dump(y, os.path.join(save_path, "y_lstm.pkl"))
joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))

print("X, y, and scaler saved successfully.")
