from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate MAPE and RMSE between actual and predicted values.
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mape, rmse