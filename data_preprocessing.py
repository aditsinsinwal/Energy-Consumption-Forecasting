import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path):
    """
    Loads the CSV file, parses the datetime column,
    fills missing values, and ensures hourly frequency.
    """
    print("Loading data...")
    df = pd.read_csv(file_path, parse_dates=['Datetime'])

    # Set datetime as index
    df.set_index('Datetime', inplace=True)
    df = df.sort_index()
    
    # Ensure data is hourly
    df = df.asfreq('H')

    # Fill missing values
    df['PJME_MW'] = df['PJME_MW'].interpolate(method='time')

    return df

def create_features(df):
    """
    Create time-based features for forecasting.
    """
    print("Creating time-based features...")

    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['is_weekend'] = df.index.dayofweek >= 5
    df['lag_24h'] = df['PJME_MW'].shift(24)
    df['rolling_mean_24h'] = df['PJME_MW'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['PJME_MW'].rolling(window=24).std()

    df.dropna(inplace=True)

    return df

def save_processed_data(df, output_path="processed_data.csv"):
    """
    Saves the processed data to a CSV file.
    """
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path)

if __name__ == "__main__":
    file_path = "PJME_hourly.csv"  

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = load_and_clean_data(file_path)
    df = create_features(df)
    save_processed_data(df)

    print("✅ Data preprocessing completed.")
