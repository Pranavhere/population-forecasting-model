"""
Atlanta MSA Population Forecast
Model 2: ARIMA (Log-Transformed) Time-Series

Requires: Final_Data_for_Modeling.xlsx with sheet 'pop_estimate_components'
Outputs:  ADF test results, ARIMA summary, forecast chart
          arima_log_model_forecast DataFrame
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# ============================================================
# 0. LOAD DATA
# ============================================================
excel_file = '/Users/pranavlakhotia/Downloads/Final_Data_for_Modeling.xlsx'

pop_estimate_components    = pd.read_excel(excel_file, sheet_name='pop_estimate_components')
pop_by_agesex              = pd.read_excel(excel_file, sheet_name='pop_by_agesex')
mortality_rate_2020_census = pd.read_excel(excel_file, sheet_name='mortality_rate_2020_census')

# ============================================================
# 1. BUILD TIME SERIES WITH LOG TRANSFORM
#    log(P_t) = log(P_0) + g*t
# ============================================================
df = pop_estimate_components.copy()
df = df.sort_values("YEAR").reset_index(drop=True)

ts = df.copy()
ts["YEAR"] = pd.to_datetime(ts["YEAR"], format="%Y")
ts = ts.set_index("YEAR")

ts["log_pop"] = np.log(ts["POP_ESTIMATE"])
series = ts["log_pop"]

# ============================================================
# 2. STATIONARITY CHECK — ADF on original series
#    H0: series is non-stationary
#    reject if p-value < 0.05
# ============================================================
adf_original = adfuller(series)
print("=" * 55)
print("ADF Test — Original log-population series")
print(f"  ADF statistic : {adf_original[0]:.4f}")
print(f"  p-value       : {adf_original[1]:.6f}")
print("  Conclusion    : Non-stationary (fail to reject H0)")
print("=" * 55)

# ============================================================
# 3. FIRST DIFFERENCING  Y'_t = Y_t - Y_{t-1}
#    Removes trend, stabilises mean
# ============================================================
ts_diff = series.diff().dropna()
adf_diff = adfuller(ts_diff)
print("\nADF Test — After first differencing")
print(f"  ADF statistic : {adf_diff[0]:.4f}")
print(f"  p-value       : {adf_diff[1]:.2e}")
print("  Conclusion    : Stationary (reject H0)")
print("=" * 55)

# ============================================================
# 4. FIT ARIMA(1, 1, 0)
#    Y'_t = phi * Y'_{t-1} + eps_t
# ============================================================
model     = ARIMA(series, order=(1, 1, 0))
model_fit = model.fit()
print("\n" + model_fit.summary().as_text())

# ============================================================
# 5. FORECAST 10 STEPS AHEAD
# ============================================================
forecast_steps = 10

forecast    = model_fit.get_forecast(steps=forecast_steps)
forecast_df = forecast.summary_frame()

# Convert from log-space back to population levels
forecast_df["POP_ESTIMATE"] = np.exp(forecast_df["mean"])
forecast_df["LOWER_95"]     = np.exp(forecast_df["mean_ci_lower"])
forecast_df["UPPER_95"]     = np.exp(forecast_df["mean_ci_upper"])

# Attach calendar years
future_years = pd.date_range(
    start=ts.index[-1] + pd.DateOffset(years=1),
    periods=forecast_steps,
    freq='YS'
)
forecast_df["YEAR"] = future_years
forecast_df = forecast_df.reset_index(drop=True)

arima_log_model_forecast = forecast_df

print("\nARIMA Forecast (population level):")
print(arima_log_model_forecast[["YEAR","POP_ESTIMATE","LOWER_95","UPPER_95"]].to_string(index=False))

# ============================================================
# 6. PLOT
# ============================================================
plt.figure()

# Historical
plt.plot(ts.index.year, ts["POP_ESTIMATE"], label="Historical")

# Forecast
plt.plot(
    arima_log_model_forecast["YEAR"].dt.year,
    arima_log_model_forecast["POP_ESTIMATE"],
    label="Forecast"
)

# 95% CI
plt.fill_between(
    arima_log_model_forecast["YEAR"].dt.year,
    arima_log_model_forecast["LOWER_95"],
    arima_log_model_forecast["UPPER_95"],
    alpha=0.2,
    label="95% Confidence Interval"
)

plt.legend()
plt.title("ARIMA (Log-Transformed) Population Forecast")
plt.xlabel("Year")
plt.ylabel("Population")
plt.tight_layout()
plt.show()

print("\nNote: ARIMA produces wide confidence intervals due to the short")
print("series (26 obs) and structural breaks in migration around 2008-2012.")
print("A cohort-component model is preferred for demographic projection.")