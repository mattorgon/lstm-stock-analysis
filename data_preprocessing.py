# data_preprocessing.py
from data_fetcher import fetch_stock_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(ticker, start_date, end_date, seq_length):
    # Fetch historical stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # Normalize the data
    data = stock_data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

    # Create sequences
    train_sequences = create_sequences(train_data, seq_length)
    test_sequences = create_sequences(test_data, seq_length)

    return train_sequences, test_sequences, scaler

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append((seq, label))
    return sequences
