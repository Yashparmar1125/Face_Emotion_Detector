import yfinance as yf
import pandas_ta as ta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

# Suppress FutureWarnings
def predict(ticker):
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Step 1: Download Data (or you can use your preloaded data)
    # ticker = 'BEL.NS'  # Change this to the stock you're interested in
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

    # Step 9: Manually Tuning Hyperparameters
    best_score = float('inf')
    best_params = {}

    # Hyperparameter grid to manually tune
    n_estimators = [50, 100, 200]
    learning_rate = [0.01, 0.1, 0.2]
    max_depth = [3, 5, 7]
    subsample = [0.7, 0.8, 1.0]
    colsample_bytree = [0.7, 0.8, 1.0]
    min_child_weight = [1, 3, 5]

    # Loop through all combinations of hyperparameters
    for n_est in n_estimators:
        for lr in learning_rate:
            for depth in max_depth:
                for ss in subsample:
                    for col in colsample_bytree:
                        for min_child in min_child_weight:
                            model = XGBRegressor(
                                n_estimators=n_est, learning_rate=lr,
                                max_depth=depth, subsample=ss,
                                colsample_bytree=col, min_child_weight=min_child
                            )
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)

                            # Update best parameters based on performance
                            if mse < best_score:
                                best_score = mse
                                best_params = {
                                    'n_estimators': n_est,
                                    'learning_rate': lr,
                                    'max_depth': depth,
                                    'subsample': ss,
                                    'colsample_bytree': col,
                                    'min_child_weight': min_child
                                }

    # Print the best hyperparameters
    print(f"Best Parameters: {best_params}")
    print(f"Best MSE: {best_score:.2f}")

    # Step 10: Evaluate the Model with Best Hyperparameters
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # Step 11: Evaluate the Best Model
    y_pred = best_model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Step 12: Predicting the next day's stock price
    last_data_point = data.iloc[-1][features].values.reshape(1, -1)
    last_data_point_scaled = scaler.transform(last_data_point)

    predicted_price = best_model.predict(last_data_point_scaled)
    print(f"Predicted next day's closing price: {predicted_price[0]:.2f}")

    # Step 13: Plotting Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Price', color='red')
    plt.legend()
    plt.title(f"Stock Prediction for {ticker}")
    plt.show()

    # Step 14: Predicting Future Stock Prices (Next 10 days)
    print(f"\nPredicting the next 10 days of stock prices:")

    # Initialize the feature vector for prediction
    last_data_point = data.iloc[-1][features].values.reshape(1, -1)
    last_data_point_scaled = scaler.transform(last_data_point)

    # Create a new list to store the predictions
    predictions = []

    for day in range(1, 11):
        prediction = best_model.predict(last_data_point_scaled)
        predictions.append(prediction[0])  # Append predicted price to the list

        # Update the feature vector with the predicted value (appending prediction and shifting the features)
        last_data_point_scaled = np.roll(last_data_point_scaled, shift=-1, axis=1)  # Shift the features
        last_data_point_scaled[0, -1] = prediction[0]  # Update the last feature with the predicted value

    # Print the predictions for the next 10 days
    for day, predicted_price in enumerate(predictions, start=1):
        print(f"Day {day}: Predicted Next Close Price: {predicted_price:.2f}")



def main():
    ticker=['RELIANCE.NS','TCS.NS','HDFCBANK.NS','BHARTIARTL.NS','SBIN.NS','NTPC.NS','NHPC.NS','SAIL.NS','SJVN.NS','POWERGRID.NS','BEL.NS','BHEL.NS','IDEA.NS','ASHOKLEY.NS','TITAGARH.NS','SUZLON.NS','IOC.NS','TATASTEEL.NS',]
    for i in ticker:
        predict(i)


if __name__ == "__main__":
    predict('ITI.NS')