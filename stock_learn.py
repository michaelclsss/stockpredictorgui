import yfinance as yf
import pandas as pd
from plot_predictions import plot_predictions
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import os
import ta


def predict_stock_price(ticker):
    try:
        end_date = datetime.today()
        # Takes 3 years
        start_date = end_date - timedelta(days=3 * 365)
        # Download data from yahoo
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        data = data[['Close']].copy()
        data['Close'] = data['Close'].astype(float) # Assure float data type
        data['Close_1'] = data['Close'].shift(1) # Yesterday's price
        data['Close_5'] = data['Close'].shift(5) # price 5 days ago
        data['Close_10'] = data['Close'].shift(10)
        data['MA_3'] = data['Close'].rolling(window=3).mean() # Past 3 day mean
        data['MA_7'] = data['Close'].rolling(window=7).mean()
        data['Return_1d'] = data['Close'].pct_change(1) # Daily return 
        data['Volatility_5d'] = data['Close'].rolling(window=5).std() # Standard deviation of past 5 day

        rsi = ta.momentum.RSIIndicator(close=data['Close'].squeeze(), window=14) # RSI (Relative Strength Index) with a 14-day window
        data['RSI_14'] = rsi.rsi() # It helps to identify whether a stock is overbought
        macd = ta.trend.MACD(close=data['Close'].squeeze()) # A trend-following momentum indicator, relationship between two moving averages of a stock’s price
        data['MACD'] = macd.macd()
        momentum = ta.momentum.ROCIndicator(close=data['Close'].squeeze(), window=10) # Measures the percentage change in price over a specific period
        data['Momentum_10'] = momentum.roc()

        data.dropna(inplace=True)

        X = data[['Close_1', 'Close_5', 'Close_10', 'MA_3', 'MA_7', 'Return_1d', 'Volatility_5d', 'RSI_14', 'MACD', 'Momentum_10']]
        y = data['Close'].values.ravel() # actual close price

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

        model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate rmse and mape
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")

        # Generate plot
        fig = plot_predictions(y_test, y_pred, X_test.index, ticker)
        fig.axes[0].set_title(f'{ticker} Price Prediction\nRMSE: {rmse:.2f}, MAPE: {mape:.2f}%')

        # Predict next day's price
        latest_features = data.iloc[-1][['Close_1', 'Close_5', 'Close_10', 'MA_3', 'MA_7', 'Return_1d', 'Volatility_5d', 'RSI_14', 'MACD', 'Momentum_10']].values.reshape(1, -1)
        next_day_prediction = model.predict(latest_features)[0]

        # Log to CSV
        today = datetime.today().strftime('%Y-%m-%d')
        pred_log = pd.DataFrame([[today, next_day_prediction, ticker]],columns=['Date', 'Predicted_Close', 'Stock'])
        log_file = 'predictions_log.csv'
        if os.path.exists(log_file):
            pred_log.to_csv(log_file, mode='a', header=False, index=False)
        else:
            pred_log.to_csv(log_file, index=False)

        return fig, next_day_prediction

    except Exception as e:
        raise RuntimeError(f"Error processing {ticker}: {str(e)}")