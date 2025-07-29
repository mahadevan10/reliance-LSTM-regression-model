import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from datetime import datetime, timedelta

# Load the Excel file
def load_data():
    # Download last 5000 days of Reliance Industries stock data
    stock = yf.download('RELIANCE.NS', period='5000d', interval='1d', progress=False)
    return stock

# Clean the data and round specific columns
def clean_data(stock):
    
    column_names = [ 'Close', 'High', 'Low', 'Open', 'Volume']
    stock.columns = column_names
    # Convert all values to percentage changes
    stock = stock.pct_change().dropna()
    stock.dropna(inplace=True)
    return stock

#combine the load and clean functions
def preprocess_data():
    stock = load_data()
    cleaned_stock = clean_data(stock)
    return cleaned_stock



df = preprocess_data()
print(df.tail())
print(df.columns)
