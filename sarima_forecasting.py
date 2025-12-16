import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot

def analyze_sarima(ts_data, frequency, diff_order, seasonal_diff_order, seasonal_period, sarima_order, seasonal_order, forecast_steps):
    """
    Hàm tổng hợp quy trình SARIMA: Vẽ biểu đồ -> Diff -> ACF/PACF -> Fit Model -> Forecast
    """
    
    # 1. Visualization
    plt.figure(figsize=(10, 4))
    plt.plot(ts_data, label='Observed', color='dodgerblue')
    plt.title("Time Series Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. Month Plot (Seasonal Check)
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        month_plot(ts_data, ax=ax)
        plt.title("Month Plot")
        plt.show()
    except:
        print("Skipping Month Plot due to index format issues")

    # 3. Differencing (Trend & Seasonality)
    # Loại bỏ xu hướng (Trend)
    dx = ts_data.diff(diff_order).dropna()
    plt.figure(figsize=(10, 3))
    plt.plot(dx, title=f"Difference d={diff_order} (Trend Removal)")
    plt.show()

    # Loại bỏ mùa vụ (Seasonality)
    ddx = dx.diff(seasonal_period).dropna()
    plt.figure(figsize=(10, 3))
    plt.plot(ddx, title=f"Seasonal Difference D={seasonal_diff_order}, s={seasonal_period}")
    plt.show()

    # 4. ACF & PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(ddx, ax=ax1, lags=2*seasonal_period)
    plot_pacf(ddx, ax=ax2, lags=2*seasonal_period)
    plt.suptitle("ACF and PACF of Transformed Data")
    plt.show()

    # 5. Build & Fit Model
    # order=(p,d,q), seasonal_order=(P,D,Q,s)
    print(f"\n--- Fitting SARIMA{sarima_order}x{seasonal_order}{seasonal_period} ---")
    model = SARIMAX(ts_data, 
                    order=sarima_order, 
                    seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    print(results.summary())

    # 6. Residual Analysis
    results.plot_diagnostics(figsize=(10, 8))
    plt.show()

    # 7. In-sample fit vs Observed
    predict = results.get_prediction()
    predicted_mean = predict.predicted_mean
    
    plt.figure(figsize=(12, 5))
    plt.plot(ts_data, label='Observed', color='dodgerblue')
    plt.plot(predicted_mean, label='Predicted (In-sample)', color='red', alpha=0.7)
    plt.title("Observed vs Predicted Values")
    plt.legend()
    plt.show()

    # 8. Forecast Future
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    plt.figure(figsize=(12, 5))
    plt.plot(ts_data, label='History', color='dodgerblue')
    plt.plot(forecast_mean, label='Forecast', color='green')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='green', alpha=0.1)
    plt.title(f"Forecast for next {forecast_steps} steps")
    plt.legend()
    plt.show()


df_motor = pd.read_csv("Motor-vehicle_deaths_2018-2023.csv")
# Start: 2018-01, Frequency: Month
date_range = pd.date_range(start='2018-01-01', periods=len(df_motor), freq='MS')
df_motor.index = date_range
ts_motor = df_motor['Deaths']

# Model chọn: SARIMA(0, 1, 2)x(1, 1, 1, 12)
analyze_sarima(ts_data=ts_motor, 
                frequency='M', 
                diff_order=1,            # d=1
                seasonal_diff_order=1,   # D=1
                seasonal_period=12,      # s=12
                sarima_order=(0,1,2),    # (p,d,q)
                seasonal_order=(1,1,1),  # (P,D,Q)
                forecast_steps=12)       # n.ahead=12
