import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm_model import train_test_split

# Define paths
data_path = r"E:\randomForestClassifier\reliance-LSTM-classifier\data"
model_path = r"E:\randomForestClassifier\reliance-LSTM-classifier\models\reliance_lstm_model.h5"

# Load data and model
X = joblib.load(os.path.join(data_path, "X_lstm.pkl"))
y = joblib.load(os.path.join(data_path, "y_lstm.pkl"))
model = load_model(model_path)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Predict
y_pred = model.predict(X_test).flatten()

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot: True vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', linewidth=2)
plt.plot(y_pred, label='Predicted', linewidth=2)
plt.title('LSTM Model: Actual vs Predicted Closing Prices (Test Set)')
plt.xlabel('Time Step')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))
plt.plot(residuals, label='Residuals', color='purple')
plt.axhline(0, linestyle='--', color='gray')
plt.title("Prediction Residuals")
plt.xlabel("Time Step")
plt.ylabel("Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
