import pandas as pd
import matplotlib.pyplot as plt
from utils import calculate_metrics

def plot_full_forecast(merged):
    plt.figure(figsize=(16,6))
    plt.plot(merged['Datetime'], merged['PJME_MW'], label='Actual Consumption', color='black', linewidth=2)
    plt.plot(merged['Datetime'], merged['ARIMA_Forecast'], label='ARIMA Forecast', color='blue', linestyle='--')
    plt.plot(merged['Datetime'], merged['LSTM_Forecast'], label='LSTM Forecast', color='green', linestyle='-.')
    plt.title("Full Forecast: Actual vs ARIMA vs LSTM", fontsize=14)
    plt.xlabel("Datetime")
    plt.ylabel("Energy Consumption (MW)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/full_forecast_comparison.png')
    plt.show()

def plot_zoomed_forecast(merged, days=7):
    zoomed = merged.tail(24 * days)
    plt.figure(figsize=(16,6))
    plt.plot(zoomed['Datetime'], zoomed['PJME_MW'], label='Actual Consumption', color='black', linewidth=2)
    plt.plot(zoomed['Datetime'], zoomed['ARIMA_Forecast'], label='ARIMA Forecast', color='blue', linestyle='--')
    plt.plot(zoomed['Datetime'], zoomed['LSTM_Forecast'], label='LSTM Forecast', color='green', linestyle='-.')
    plt.title(f"Zoomed Forecast (Last {days} Days): Actual vs ARIMA vs LSTM", fontsize=14)
    plt.xlabel("Datetime")
    plt.ylabel("Energy Consumption (MW)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'outputs/zoomed_forecast_last_{days}_days.png')
    plt.show()

def display_metrics_table(arima_metrics, lstm_metrics):
    metrics_df = pd.DataFrame({
        'Model': ['ARIMA', 'LSTM'],
        'MAPE (%)': [round(arima_metrics[0]*100, 2), round(lstm_metrics[0]*100, 2)],
        'RMSE': [round(arima_metrics[1], 2), round(lstm_metrics[1], 2)]
    })
    print("\n📊 Forecast Evaluation Summary:")
    print(metrics_df.to_string(index=False))

def main():
    print("Loading forecast and actual data...")
    df = pd.read_csv('processed_data.csv', parse_dates=['Datetime'])
    arima_df = pd.read_csv('outputs/arima_forecast.csv', parse_dates=['Datetime'])
    lstm_df = pd.read_csv('outputs/lstm_forecast.csv', parse_dates=['Datetime'])

    # Align actual values
    actual = df[df['Datetime'].isin(arima_df['Datetime'])][['Datetime', 'PJME_MW']]

    # Merge forecasts
    merged = actual.merge(arima_df, on='Datetime').merge(lstm_df, on='Datetime')

    # Plot full forecast
    plot_full_forecast(merged)

    # Plot zoomed-in forecast (last 7 days)
    plot_zoomed_forecast(merged, days=7)

    # Calculate metrics
    arima_metrics = calculate_metrics(merged['PJME_MW'], merged['ARIMA_Forecast'])
    lstm_metrics = calculate_metrics(merged['PJME_MW'], merged['LSTM_Forecast'])

    # Display metrics table
    display_metrics_table(arima_metrics, lstm_metrics)

if __name__ == "__main__":
    main()
