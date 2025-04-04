import yfinance as yf
import pandas_ta as ta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
import requests


def fetch_sentiment_data(ticker):
    """
    Fetch sentiment data for a given stock ticker using an external API.
    Example: You would need to use an API to fetch sentiment data.
    """
    # Example API call (replace with real API)
    sentiment_data = requests.get(f'https://api.sentimentanalysis.com/{ticker}').json()
    return sentiment_data


def calculate_technical_indicators(data, close_column, volume_column):
    """
    Adds technical indicators to the stock data.
    """
    # Flatten multi-level columns if present (e.g., for Adjusted Close)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns]

    # Check the columns to confirm correct names
    print("Columns in data:", data.columns)

    # Ensure that the 'High', 'Low', 'Close' columns are correctly identified
    close_column = [col for col in data.columns if 'Close' in col][0]  # Find the Close column
    high_column = [col for col in data.columns if 'High' in col][0]  # Find the High column
    low_column = [col for col in data.columns if 'Low' in col][0]  # Find the Low column
    volume_column = [col for col in data.columns if 'Volume' in col][0]  # Find the Volume column

    # Debugging: Print the identified columns
    print(
        f"Close column: {close_column}, High column: {high_column}, Low column: {low_column}, Volume column: {volume_column}")

    # Technical Indicator Calculations
    data['SMA'] = ta.sma(data[close_column], length=50)
    data['EMA'] = ta.ema(data[close_column], length=50)
    data['RSI'] = ta.rsi(data[close_column], length=14)

    # On-Balance Volume calculation with error handling
    try:
        data['OBV'] = ta.obv(data[close_column], data[volume_column])
    except Exception as e:
        print(f"Error calculating OBV: {e}")

    # Bollinger Bands calculation
    try:
        bollinger_bands = ta.bbands(data[close_column], length=50)
        data['Bollinger_Upper'] = bollinger_bands['BBU_50_2.0']
        data['Bollinger_Middle'] = bollinger_bands['BBM_50_2.0']
        data['Bollinger_Lower'] = bollinger_bands['BBL_50_2.0']
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")

    # Add MACD
    data['MACD'] = ta.macd(data[close_column])['MACD_12_26_9']

    # Add ADX (using High, Low, Close)
    data['ADX'] = ta.adx(data[high_column], data[low_column], data[close_column])['ADX_14']

    # Add ATR (Average True Range)
    data['ATR'] = ta.atr(data[high_column], data[low_column], data[close_column], length=14)

    # Add MFI (Money Flow Index)
    data['MFI'] = ta.mfi(data[high_column], data[low_column], data[close_column], data[volume_column], length=14)

    return data


def add_lag_features(data, close_column, lag_days=5):
    """
    Adds lag features for stock prices.
    """
    if close_column not in data.columns:
        raise KeyError(f"The column '{close_column}' doesn't exist in the dataset.")

    for i in range(1, lag_days + 1):
        data[f'Lag_{i}_Close'] = data[close_column].shift(i)

    return data


def backtest_strategy(data, model, features, target_column):
    """
    Backtest the model using a simple strategy.
    """
    predictions = model.predict(data[features])
    data['Predicted_Close'] = predictions

    # Create a simple strategy: Buy when the predicted price goes up and sell when it goes down
    data['Signal'] = np.where(data['Predicted_Close'] > data['Close'], 1, 0)  # 1 = Buy, 0 = Sell

    # Calculate returns
    data['Return'] = data['Close'].pct_change() * data['Signal'].shift(1)  # Simple return
    print(f"Total return: {data['Return'].sum()}")


def rolling_window_cv(X, y, n_splits=5):
    """
    Performs Rolling Window Cross-Validation.
    """
    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in ts_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f'MSE: {mean_squared_error(y_test, y_pred)}')
        print(f'R2: {r2_score(y_test, y_pred)}')


def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using various metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")


def predict(ticker):
    # Step 1: Download Data
    data = yf.download(ticker, start="2000-01-01", end="2025-01-10")

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns]

    # Step 2: Dynamically identify the 'Close' and 'Volume' columns
    try:
        close_column = [col for col in data.columns if 'Close' in col][0]  # Dynamically find 'Close'
        volume_column = [col for col in data.columns if 'Volume' in col][0]  # Dynamically find 'Volume'
    except IndexError as e:
        print(f"Error: Required columns ('Close or Volume') not found in the data for {ticker}.")
        return

    # Step 3: Feature Engineering
    data = calculate_technical_indicators(data, close_column, volume_column)
    data = add_lag_features(data, close_column)  # Pass the identified close_column

    # Drop NaN rows after feature engineering
    data = data.dropna()

    # Step 4: Prepare the Target Variable (Next day's Close Price)
    data['Next_Close'] = data[close_column].shift(-1)  # Predict the next day's closing price
    data = data.dropna()  # Drop the last row (NaN target)

    # Step 5: Prepare Features and Target
    features = ['SMA', 'EMA', 'RSI', 'OBV', 'Bollinger_Upper', 'Bollinger_Lower', 'MACD', 'ADX', 'ATR', 'MFI']
    features += [f'Lag_{i}_Close' for i in range(1, 6)]  # Include lag features
    X = data[features]
    y = data['Next_Close']

    # Step 6: Normalize the Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 7: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 8: Model Training with Hyperparameter Tuning
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    # Step 9: Model Evaluation
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

    # Step 10: Predicting the next day's stock price
    last_data_point = data.iloc[-1][features].values.reshape(1, -1)
    last_data_point_scaled = scaler.transform(last_data_point)

    predicted_price = model.predict(last_data_point_scaled)
    print(f"Predicted next day's closing price for {ticker}: {predicted_price[0]:.2f}")

    # Step 11: Plotting Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Price', color='red')
    plt.legend()
    plt.title(f"Stock Prediction for {ticker}")
    plt.show()

    # Step 12: Backtesting the Strategy
    backtest_strategy(data, model, features, 'Next_Close')

    # Step 13: Rolling Window Cross-Validation
    rolling_window_cv(X_scaled, y, n_splits=5)


def main():
    tickers = [ 'BHEL.NS']
    for ticker in tickers:
        predict(ticker)


if __name__ == "__main__":
    main()
