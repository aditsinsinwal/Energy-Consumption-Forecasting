import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_percentage_error

def create_lstm_dataset(series, look_back=24):
    X, Y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        Y.append(series[i+look_back])
    return np.array(X), np.array(Y)

def run_lstm_model(data_path):
    print("Loading processed data...")
    df = pd.read_csv(data_path, parse_dates=['Datetime'], index_col='Datetime')
    values = df['PJME_MW'].values.reshape(-1,1)

    # Scale data
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    look_back = 24
    X, y = create_lstm_dataset(scaled_values, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split data (Train: 90%, Test: 10%)
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("Building LSTM model...")
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    print("Training LSTM...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Predict
    predictions = model.predict(X_test)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

    # Evaluate
    mape = mean_absolute_percentage_error(y_test_inv, predictions_inv)
    print(f"LSTM MAPE: {mape:.2%}")

    # Save forecast
    forecast_df = pd.DataFrame({
        'Datetime': df.index[-len(predictions_inv):],
        'LSTM_Forecast': predictions_inv.flatten()
    })
    forecast_df.to_csv('outputs/lstm_forecast.csv', index=False)

    print("✅ LSTM forecast saved to outputs/lstm_forecast.csv")

if __name__ == "__main__":
    run_lstm_model('processed_data.csv')
