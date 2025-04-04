import yfinance as yf
import pandas_ta as ta
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def predict(ticker):
    # Step 1: Download Data (or you can use your preloaded data)
    # ticker = 'SALASAR.NS'  # Change this to the stock you're interested in
    data = yf.download(ticker, start="2000-01-01", end="2025-01-10")
    data2 = yf.download(ticker, start="2000-01-01", end="2025-01-11")
    print(data2)


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

    # Step 8: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 9: Train XGBoost Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    # Step 10: Evaluate the Model
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    smse=np.sqrt(mse)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.2f}")
    print(f"R-squared: {r2:.2f}")

    # Step 11: Predicting the next day's stock price
    last_data_point = data.iloc[-1][features].values.reshape(1, -1)
    last_data_point_scaled = scaler.transform(last_data_point)

    predicted_price = model.predict(last_data_point_scaled)
    print(f"Predicted next day's closing price for {ticker}: {predicted_price[0]:.2f}")
    print("---------------------------------------------------------------------------------------------------")

    # Step 12: Plotting Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Price', color='red')
    plt.legend()
    plt.title(f"Stock Prediction for {ticker}")
    plt.show()

    # Step 13: Predicting Future Stock Prices (Next 10 days)
    # Step 13: Predicting Future Stock Prices (Next 10 days)
    print(f"\nPredicting the next 10 days of stock prices:")

    # Initialize the feature vector for prediction
    last_data_point = data.iloc[-1][features].values.reshape(1, -1)
    last_data_point_scaled = scaler.transform(last_data_point)

    # Create a new list to store the predictions
    predictions = []

    for day in range(1, 11):
        prediction = model.predict(last_data_point_scaled)
        predictions.append(prediction[0])  # Append predicted price to the list

        # Update the feature vector with the predicted value (appending prediction and shifting the features)
        last_data_point_scaled = np.roll(last_data_point_scaled, shift=-1, axis=1)  # Shift the features
        last_data_point_scaled[0, -1] = prediction[0]  # Update the last feature with the predicted value

    # Print the predictions for the next 10 days
    for day, predicted_price in enumerate(predictions, start=1):
        print(f"Day {day}: Predicted Next Close Price: {predicted_price:.2f}")


def main():
    ticker=['RELIANCE.NS','ITI.NS','TCS.NS','HDFCBANK.NS','BHARTIARTL.NS','SBIN.NS','NTPC.NS','NHPC.NS','SAIL.NS','SJVN.NS','POWERGRID.NS','BEL.NS','BHEL.NS','IDEA.NS','ASHOKLEY.NS','TITAGARH.NS','SUZLON.NS','IOC.NS','TATASTEEL.NS','RKFORGE.NS',]
    for i in ticker:
        predict(i)


if __name__ == "__main__":
    predict('BEL.NS')

