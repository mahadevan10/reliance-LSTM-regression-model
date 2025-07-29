# LSTM model definition and training for Reliance LSTM project


import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os


# Define path
path = r"E:\randomForestClassifier\reliance-LSTM-classifier\data"

# Load preprocessed sequences
X = joblib.load(f"{path}\X_lstm.pkl")
y = joblib.load(f"{path}\y_lstm.pkl")

# Check shapes
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


#train test split with respect to time series
def train_test_split(X, y):
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Define the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)  # No activation, since it's regression
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# Early stopping feature
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=30,
                    batch_size=32,
                    callbacks=[early_stop])

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Define model save path
model_save_path = r"E:\randomForestClassifier\reliance-LSTM-classifier\models"
os.makedirs(model_save_path, exist_ok=True)

# Save the model
model.save(os.path.join(model_save_path, "reliance_lstm_model.h5"))

print("Model saved successfully.")