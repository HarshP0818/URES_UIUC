import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

XLE_data = pd.read_csv(r"C:/Users/harsh/OneDrive/Desktop/XLE.csv", index_col='Date', parse_dates=True)

close_data = XLE_data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size:, :]

window_size = 30
X_train = []
y_train = []
for i in range(window_size, len(train_data)):
    X_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1), order='C')

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss = 'mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, batch_size=1, epochs=15)

X_test = []
y_test = close_data[train_size+window_size:]
if len(test_data) >= window_size:
    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-window_size:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

mse = np.mean(((predictions - y_test) ** 2))
mae = np.mean(np.abs(predictions - y_test))
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', np.sqrt(mse))
print('Mean Absolute Error (MAE):', mae)
print('Mean Absolute Percentage Error (MAPE):', mape, '%')
print('R-squared (R2):', r2)

plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('SPY Stock Price Forecast')
plt.legend()

plt.show()

forecast_horizon = 7
forecast_input = test_data[-window_size:, :]
forecast = []

for i in range(forecast_horizon):
    # Reshape the forecast input data for the model
    x = forecast_input[-window_size:]
    x = np.reshape(x, (1, window_size, 1))

    # Generate the next day's prediction
    pred = model.predict(x)[0][0]

    # Append the prediction to the forecast list and the forecast input data
    forecast.append(pred)
    forecast_input = np.append(forecast_input, pred)
    forecast_input = np.reshape(forecast_input, (-1, 1))

# Scale back the forecast values
forecast = np.array(forecast).reshape(-1, 1)
forecast = scaler.inverse_transform(forecast)

# Plot the forecast values
last_date = XLE_data.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='B')[1:]
plt.plot(forecast_dates, forecast, label='Forecast')
plt.legend()
plt.show()
