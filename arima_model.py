import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

def run_arima_model(data_path):
    print("Loading processed data...")
    df = pd.read_csv(data_path, parse_dates=['Datetime'], index_col='Datetime')

    # Use only the target variable
    y = df['PJME_MW']

    # Split data (Train: 80%, Test: 20%)
    split_idx = int(len(y) * 0.8)
    train, test = y[:split_idx], y[split_idx:]

    print("Fitting ARIMA model...")
    model = ARIMA(train, order=(5,1,2))   # You can tune this order
    model_fit = model.fit()

    print("Forecasting...")
    forecast = model_fit.forecast(steps=len(test))

    # Evaluate
    mape = mean_absolute_percentage_error(test, forecast)
    print(f"ARIMA MAPE: {mape:.2%}")

    # Save forecast
    forecast_df = pd.DataFrame({'Datetime': test.index, 'ARIMA_Forecast': forecast.values})
    forecast_df.to_csv('outputs/arima_forecast.csv', index=False)

    print("✅ ARIMA forecast saved to outputs/arima_forecast.csv")

if __name__ == "__main__":
    run_arima_model('processed_data.csv')
