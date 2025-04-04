import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Suppress FutureWarning for XGBRegressor
warnings.filterwarnings("ignore", category=FutureWarning, message=".*XGBRegressor.*")


def predict(ticker):
    # Step 1: Download Data (or you can use your preloaded data)
    data = yf.download(ticker, start="2000-01-01", end="2025-01-10")

    # Flatten multi-level columns if present
    data.columns = ['_'.join(col).strip() for col in data.columns]

    # Debugging: Print the columns to understand the structure
    print("Columns after flattening:", data.columns)

    # Step 2: Dynamically identify the 'Close' and 'Volume' columns
    close_column = [col for col in data.columns if 'Close' in col][0]  # Find the Close column
    volume_column = [col for col in data.columns if 'Volume' in col][0]  # Find the Volume column

    # Step 2: Validate that data has the necessary columns for analysis
    required_columns = [close_column, volume_column]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Data must contain the following columns: {', '.join(required_columns)}")

    # Step 3: Check for missing values and drop rows with NaN values
    data = data.dropna(subset=[close_column, volume_column])

    # Step 4: Feature Engineering - Calculate Technical Indicators using pandas_ta
    data['SMA'] = ta.sma(data[close_column], length=50)  # Simple Moving Average
    data['EMA'] = ta.ema(data[close_column], length=50)  # Exponential Moving Average
    data['RSI'] = ta.rsi(data[close_column], length=14)  # Relative Strength Index
    data['MACD'] = ta.macd(data[close_column])['MACD_12_26_9']  # Moving Average Convergence Divergence

    # OBV calculation with error handling
    try:
        data['OBV'] = ta.obv(data[close_column], data[volume_column])  # On-Balance Volume
    except Exception as e:
        print(f"Error calculating OBV: {e}")

    # Bollinger Bands calculation
    try:
        bollinger_bands = ta.bbands(data[close_column], length=50)
        data['Bollinger_Upper'] = bollinger_bands['BBU_50_2.0']  # Adjust the column name to match the DataFrame output
        data['Bollinger_Middle'] = bollinger_bands['BBM_50_2.0']
        data['Bollinger_Lower'] = bollinger_bands['BBL_50_2.0']
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")

    # Drop any rows with missing values after indicator calculation
    data = data.dropna()

    # Step 5: Prepare the Target Variable (Next day's Close Price)
    data['Next_Close'] = data[close_column].shift(-1)  # Predict the next day's closing price
    data = data.dropna()  # Drop the last row (NaN target)

    # Step 6: Prepare Features and Target with Lagged Variables
    data['Prev_Close'] = data[close_column].shift(1)
    features = ['SMA', 'EMA', 'RSI', 'OBV', 'Bollinger_Upper', 'Bollinger_Lower', 'MACD', 'Prev_Close']
    X = data[features]
    y = data['Next_Close']

    # Step 7: Normalize the Features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 8: Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)

    # Step 9: Train XGBoost Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_scaled, y)

    # Step 10: Evaluate the Model
    y_pred = model.predict(X_scaled)

    # Evaluate model performance using multiple metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Step 11: Predicting the next day's stock price
    last_data_point = data.iloc[-1][features].values.reshape(1, -1)
    last_data_point_scaled = scaler.transform(last_data_point)

    predicted_price = model.predict(last_data_point_scaled)
    print(f"Predicted next day's closing price for {ticker}: {predicted_price[0]:.2f}")
    print("---------------------------------------------------------------------------------------------------")

    # Step 12: Plotting Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, y, label='Actual Price', color='blue')
    plt.plot(data.index, y_pred, label='Predicted Price', color='red')
    plt.legend()
    plt.title(f"Stock Prediction for {ticker}")
    plt.show()


def main():
    ticker=['RELIANCE.NS','TCS.NS','HDFCBANK.NS','BHARTIARTL.NS','SBIN.NS','NTPC.NS','NHPC.NS','SAIL.NS','SJVN.NS','POWERGRID.NS','BEL.NS','BHEL.NS','IDEA.NS','ASHOKLEY.NS','TITAGARH.NS','SUZLON.NS','IOC.NS','TATASTEEL.NS','RKFORGE.NS',]
    for i in ticker:
        predict(i)

if __name__ == '__main__':
    main()