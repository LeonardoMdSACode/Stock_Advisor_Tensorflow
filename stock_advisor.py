import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
import yfinance as yf
import mplfinance as mpf
import tensorflow as tf

stock = 'GBPUSD=X'
df = yf.download(stock)
# df.to_csv('stock_data.csv')

print(df)
print(df.shape)
print(df.describe())
print(df.isnull().sum())

# Define the plot style and type
mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
s = mpf.make_mpf_style(marketcolors=mc)
kwargs = dict(type='candle')

# Plot the data as a candlestick chart
# mpf.plot(df, **kwargs, style=s, title='GBP/USD Exchange Rate')

df = df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

print(df)
print(df.shape)
print(df.describe())
print(df.isnull().sum())

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Split the data into training and test sets
# train_data = df[:'2022']
# test_data = df['2022':]

# Normalize the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define the function to create time series data


def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)


window_size = 30
X_train, y_train = create_dataset(train_data, window_size)
X_test, y_test = create_dataset(test_data, window_size)


# Showing the graph where training and test data divide
train_size = len(X_train)
test_size = len(X_test)

plt.figure(figsize=(10, 5))
plt.title('GBP/USD Exchange Rate')
plt.plot(range(train_size), y_train, label='Training Data')
plt.plot(range(train_size, train_size + test_size), y_test, label='Test Data')
plt.xlabel('Time')
plt.ylabel('Scaled Closing Price')
plt.legend()

plt.axvline(X_train.shape[0], color='k', linestyle='--')

# plt.show()

# Reshape y_train and y_test to be 2D arrays
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# LSTM RNN
print("LSTM RNN Model")
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)


class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        units=32, return_sequences=True), input_shape=[window_size, 1]),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(units=32, return_sequences=False)),
    tf.keras.layers.Dense(units=1)
])

model.summary()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
reset_states = ResetStatesCallback()
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "my_checkpoint.h5", save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=50)
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, model_checkpoint, reset_states])

model = tf.keras.models.load_model("my_checkpoint.h5")

# Make predictions on the test set
y_pred = model.predict(X_test)
# Make predictions on the train set
y_train_pred = model.predict(X_train)

# Calculate the RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print('Train Score: %.5f RMSE' % train_rmse)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Test Score: %.5f RMSE' % test_rmse)
# Train Score: 0.01391 RMSE
# Test Score: 0.01164 RMSE


figure, axes = plt.subplots(figsize=(15, 6))
axes.xaxis_date()

axes.plot(df[len(df)-len(y_test):].index, y_test,
          color='red', label='Real GBP/USD Stock Price')
axes.plot(df[len(df)-len(y_test):].index, y_pred,
          color='blue', label='Predicted GBP/USD Stock Price')

plt.title('GBP/USD Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('GBP/USD Stock Price')
plt.legend()
plt.show()

print(df)


# Predict the behavior Close price in the next 30 days
last_30_days = test_data[-30:]

# Create an empty list to store the predicted prices
predicted_prices = []

# Predict the next day's price for 30 days
for i in range(30):
    # Reshape the data to be 3D
    X_test_new = np.reshape(last_30_days, (1, window_size, 1))
    # Make the prediction
    y_pred_new = model.predict(X_test_new)
    # Append the predicted price to the list
    predicted_prices.append(y_pred_new[0, 0])
    # Shift the window by one day and append the predicted price to last_30_days
    last_30_days = np.append(last_30_days[1:], y_pred_new)

# Convert the predicted prices back to their original scale
predicted_prices = scaler.inverse_transform(
    np.array(predicted_prices).reshape(-1, 1))

# Create a date range for the next 30 days
last_date = df.index[-1]
date_range = pd.date_range(last_date, periods=31, freq='D')[1:]

# Create a DataFrame for the predicted prices
predicted_df = pd.DataFrame(predicted_prices, columns=[
                            'Close'], index=date_range)

# Append the predicted prices to the original DataFrame
df_merged = pd.concat([df, predicted_df])

# df_merged.to_csv('stock_data_predicted.csv')
print(df_merged.tail(31))


# Showcase the predicted future prices
figure, axes = plt.subplots(figsize=(15, 6))
axes.xaxis_date()

axes.plot(df[len(df)-90:].index, df[len(df)-90:]['Close'],
          color='red', label='Real GBP/USD Stock Price')
axes.plot(df_merged[len(df_merged)-30:].index, df_merged[len(df_merged)-30:]['Close'],
          color='blue', label='Predicted GBP/USD Stock Price')

plt.title('GBP/USD Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('GBP/USD Stock Price')
plt.legend()
plt.show()
