import yfinance as yf
import pandas_ta as ta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def predict(ticker):
    # Step 1: Download Data (or you can use your preloaded data)
    data = yf.download(ticker, start="2000-01-01", end="2022-01-01")

    # Flatten multi-level columns if present
    data.columns = ['_'.join(col).strip() for col in data.columns]

    # Step 2: Dynamically identify the 'Close' and 'Volume' columns
    close_column = [col for col in data.columns if 'Close' in col][0]  # Find the Close column
    volume_column = [col for col in data.columns if 'Volume' in col][0]  # Find the Volume column

    # Step 3: Check for missing values and drop rows with NaN values
    data = data.dropna(subset=[close_column, volume_column])

    # Step 4: Feature Engineering - Calculate Technical Indicators using pandas_ta
    data['SMA'] = ta.sma(data[close_column], length=50)  # Simple Moving Average
    data['EMA'] = ta.ema(data[close_column], length=50)  # Exponential Moving Average
    data['RSI'] = ta.rsi(data[close_column], length=14)  # Relative Strength Index

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

    # Step 6: Prepare Features and Target
    features = ['SMA', 'EMA', 'RSI', 'OBV', 'Bollinger_Upper', 'Bollinger_Lower']
    X = data[features]
    y = data['Next_Close']

    # Step 7: Normalize the Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 8: Train-Test Split (Backtesting)
    train_data = data[:'2020-12-31']
    test_data = data['2021-01-01':]

    # Train XGBoost Model on the Training Set
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    X_train = train_data[features]
    y_train = train_data['Next_Close']
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)

    # Make Predictions on the Test Set
    X_test = test_data[features]
    y_test = test_data['Next_Close']
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # Evaluate model performance on the test set
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Step 9: Paper Trading Simulation
    initial_balance = 100000  # Starting balance in currency
    cash_balance = initial_balance
    stock_balance = 0  # No stocks initially

    profits = []

    for i in range(1, len(y_pred)):
        # Buy signal: if the model predicts an increase in price
        if y_pred[i] > y_pred[i-1]:
            # Buy stock
            stock_balance += cash_balance / test_data[close_column].iloc[i]
            cash_balance = 0  # Spend all cash on stocks
        elif y_pred[i] < y_pred[i-1] and stock_balance > 0:
            # Sell all stocks
            cash_balance += stock_balance * test_data[close_column].iloc[i]
            stock_balance = 0  # All stocks sold

        # Track portfolio value (cash + stock value)
        portfolio_value = cash_balance + stock_balance * test_data[close_column].iloc[i]
        profits.append(portfolio_value)

    final_balance = profits[-1]
    profit = final_balance - initial_balance
    print(f"Total simulated profit from paper trading: {profit:.2f}")

    # Step 10: Visualizing Paper Trading Performance
    plt.plot(test_data.index[1:], profits, label="Portfolio Value", color='green')
    plt.title(f"Simulated Portfolio Performance for {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

# Run the prediction and backtesting for a given stock
predict('BEL.NS')
