# lstm_stock_prediction.py
import numpy as np
from data_preprocessing import preprocess_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import yfinance as yf  # Assuming you have this library for fetching stock data

def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data['Close'].values

def plot_predictions_vs_actual(actual_prices, predicted_prices):
    # Create a time axis
    time_points = range(len(actual_prices))

    # Plot actual prices
    plt.plot(time_points, actual_prices, label='Actual Prices', color='blue')

    # Plot predicted prices
    plt.plot(time_points, predicted_prices, label='Predicted Prices', color='red', linestyle='dashed')

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Predicted Stock Prices')

    # Add a legend
    plt.legend()

    # Show or save the plot
    plt.show()

def main():
    # Specify parameters
    ticker_symbol = "SPY"
    start_date = "2020-01-01"
    end_date = "2022-01-01"
    seq_length = 20  # Adjust as needed

    # Fetch actual stock prices
    actual_prices = fetch_stock_data(ticker_symbol, start_date, end_date)

    # Preprocess data
    train_sequences, test_sequences, scaler = preprocess_data(ticker_symbol, start_date, end_date, seq_length)

    # Define the LSTM model (unchanged)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model (unchanged)
    X_train, y_train = zip(*train_sequences)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Make predictions
    X_test, y_test = zip(*test_sequences)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    predictions = model.predict(X_test)

    # Inverse transform predictions to the original scale
    predictions_original_scale = scaler.inverse_transform(predictions)

    # Plot predictions vs actual prices
    plot_predictions_vs_actual(actual_prices[-len(predictions_original_scale):], predictions_original_scale)

if __name__ == "__main__":
    main()
