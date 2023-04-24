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
keras = tf.keras

df = yf.download("GBPUSD=X", start="1990-01-01")
# df.to_csv('GBPUSD.csv')

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

scaler = MinMaxScaler(feature_range=(-1, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# function to create train, test data given stock data and sequence length


def load_data(stock, look_back):
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    test_set_size = int(np.round(0.15*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


look_back = 75
x_train, y_train, x_test, y_test = load_data(df, look_back)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

# make training and test sets in tensorflow
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

y_train.shape, x_train.shape

# Showing the graph where training and test data divide
train_size = len(x_train)
test_size = len(x_test)

plt.figure(figsize=(10, 5))
plt.title('GBP/USD Exchange Rate')
plt.plot(range(train_size), y_train, label='Training Data')
plt.plot(range(train_size, train_size + test_size), y_test, label='Test Data')
plt.xlabel('Time')
plt.ylabel('Scaled Closing Price')
plt.legend()

plt.axvline(x_train.shape[0], color='k', linestyle='--')

# plt.show()

# LSTM RNN
print("LSTM RNN Model")
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

class ResetStatesCallback(keras.callbacks.Callback):
   def on_epoch_begin(self, epoch, logs):
      self.model.reset_states()

model = keras.models.Sequential([
    keras.layers.Bidirectional(keras.layers.LSTM(units=32, return_sequences=True), input_shape=[None, 1]),
    keras.layers.Bidirectional(keras.layers.LSTM(units=32, return_sequences=True)),
    keras.layers.Dense(units=1)
])

model.summary()
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
reset_states = ResetStatesCallback()
model_checkpoint = keras.callbacks.ModelCheckpoint(
   "my_checkpoint2.h5", save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=50)
model.fit(x_train, y_train, epochs=100,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping, model_checkpoint, reset_states])

y_train_pred = model(x_train)
y_test_pred = model(x_test)

y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())
trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
print('Train Score: %.5f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
print('Test Score: %.5f RMSE' % (testScore))

model = keras.models.load_model("my_checkpoint2.h5")

figure, axes = plt.subplots(figsize=(15, 6))
axes.xaxis_date()

axes.plot(df[len(df)-len(y_test):].index, y_test,
          color='red', label='Real GBP/USD Stock Price')
axes.plot(df[len(df)-len(y_test):].index, y_test_pred,
          color='blue', label='Predicted GBP/USD Stock Price')

plt.title('GBP/USD Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('GBP/USD Stock Price')
plt.legend()
plt.show()

print(df)