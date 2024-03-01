# lstm_stock_prediction.py
import numpy as np
from data_preprocessing import preprocess_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def main():
    # Specify parameters
    ticker_symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2022-01-01"
    seq_length = 10  # Adjust as needed

    # Preprocess data
    train_sequences, test_sequences, scaler = preprocess_data(ticker_symbol, start_date, end_date, seq_length)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
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

    print(predictions_original_scale)

if __name__ == "__main__":
    main()
