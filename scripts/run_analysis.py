scripts/run_analysis.py
influenza-msar-saudi/
│
├── data/
│   └── flunet_ksa.csv
│
├── scripts/
│   ├── run_analysis.py
│
├── results/
│   ├── tables/
│   ├── figures/
│
├── models/
├── requirements.txt
└── README.md
  # =========================================
# Influenza MS-AR Pipeline (Python Version)
# =========================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Prophet
from prophet import Prophet

# LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# reproducibility
np.random.seed(42)

# =========================================
# 1. LOAD DATA
# =========================================

data = pd.read_csv("data/flunet_ksa.csv")

# clean
data = data[data["SPEC_PROCESSED_NB"] > 0]
data["positivity_rate"] = data["INF_ALL"] / data["SPEC_PROCESSED_NB"]

data["week"] = pd.to_datetime(data["week"])
data = data.sort_values("week")

# =========================================
# 2. TRAIN / TEST SPLIT
# =========================================

train = data[data["week"] <= "2023-12-31"]
test  = data[data["week"] > "2023-12-31"]

y_train = train["positivity_rate"].values
y_test  = test["positivity_rate"].values

# =========================================
# 3. AR(2) MODEL
# =========================================

ar_model = AutoReg(y_train, lags=2).fit()
ar_pred = ar_model.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)

# =========================================
# 4. SARIMA
# =========================================

sarima = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,52)).fit(disp=False)
sarima_pred = sarima.forecast(len(y_test))

# =========================================
# 5. PROPHET
# =========================================

df_prophet = train[["week","positivity_rate"]].rename(columns={"week":"ds","positivity_rate":"y"})

model_prophet = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05)
model_prophet.fit(df_prophet)

future = model_prophet.make_future_dataframe(periods=len(y_test), freq='W')
forecast = model_prophet.predict(future)

prophet_pred = forecast["yhat"].iloc[-len(y_test):].values

# =========================================
# 6. LSTM
# =========================================

def create_sequences(data, window=12):
    X, y = [], []
    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

window = 12
X_train, y_train_lstm = create_sequences(y_train, window)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(32, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train_lstm, epochs=50, batch_size=32, verbose=0)

# forecasting
lstm_pred = []
input_seq = y_train[-window:].reshape(1, window, 1)

for _ in range(len(y_test)):
    pred = model_lstm.predict(input_seq, verbose=0)[0][0]
    lstm_pred.append(pred)
    input_seq = np.append(input_seq[:,1:,:], [[[pred]]], axis=1)

lstm_pred = np.array(lstm_pred)

# =========================================
# 7. METRICS
# =========================================

def compute_metrics(y, yhat):
    rmse = np.sqrt(mean_squared_error(y, yhat))
    mae  = mean_absolute_error(y, yhat)
    return rmse, mae

results = []

models = {
    "AR(2)": ar_pred,
    "SARIMA": sarima_pred,
    "Prophet": prophet_pred,
    "LSTM": lstm_pred
}

for name, pred in models.items():
    rmse, mae = compute_metrics(y_test, pred)
    results.append([name, rmse, mae])

df_results = pd.DataFrame(results, columns=["Model","RMSE","MAE"])

# =========================================
# 8. SAVE RESULTS
# =========================================

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

df_results.to_csv("results/tables/table4.csv", index=False)

# =========================================
# 9. PLOT
# =========================================
plt.figure(figsize=(12,6))
plt.plot(test["week"], y_test, label="Actual")
plt.plot(test["week"], ar_pred, label="AR(2)")
plt.plot(test["week"], sarima_pred, label="SARIMA")
plt.plot(test["week"], prophet_pred, label="Prophet")
plt.plot(test["week"], lstm_pred, label="LSTM")
plt.legend()
plt.title("Forecast Comparison")
plt.savefig("results/figures/forecast.png")
plt.close()
print("Pipeline completed successfully.")
pandas
numpy
matplotlib
statsmodels
scikit-learn
prophet
tensorflow
pip install -r requirements.txt
python scripts/run_analysis.py
