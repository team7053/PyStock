import datetime as dt
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import urllib.request, json
import os

api_key = 'TFDMPJ42XYWICWP2'

# American Airlines stock market prices
ticker = "AAL"

# JSON file with all the stock market data for AAL from the last 20 years
url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

# Save data to this file
file_to_save = r'C:\Users\Nathan\Desktop\PyStock\stock_market_data-%s.csv'%ticker

if os.path.exists(file_to_save):
    os.remove(file_to_save)

# grab the data from the url
# And store date, low, high, volume, close, open values to a Pandas DataFrame
with urllib.request.urlopen(url_string) as url:
    data = json.loads(url.read().decode())
    # extract stock market data
    data = data['Time Series (Daily)']
    df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
    for k,v in data.items():
        date = dt.datetime.strptime(k, '%Y-%m-%d')
        data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                    float(v['4. close']),float(v['1. open'])]
        df.loc[-1,:] = data_row
        df.index = df.index + 1
print('Data saved to : %s'%file_to_save)        
df.to_csv(file_to_save)
df_sorted = df.sort_values('Date')
high_prices = df.loc[:,'High'].values
low_prices = df.loc[:,'Low'].values
mid_prices = (high_prices+low_prices)/2.0
df['Mid'] = mid_prices
test_set = mid_prices[:767]
train_set = mid_prices[767:]
mid_prices = mid_prices.reshape(-1,1)

train_set = train_set.reshape(-1,1)
test_set = test_set.reshape(-1,1)

sc = MinMaxScaler(feature_range=(0,1))
train_set_scaled = sc.fit_transform(train_set)
print(train_set_scaled)
X_train = []
y_train = []
for i in range(60, 3000):
    X_train.append(train_set_scaled[i-60:i, 0])
    y_train.append(train_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 75, batch_size = 64)


dataset_total = df['Mid']
inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1,1) 
inputs = sc.transform(inputs)
X_test = []
for i in range(60,767):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price = predicted_stock_price[::-1]
df_midprices = pd.DataFrame(data=mid_prices)



plt.style.use('classic')
plt.plot(range(df_sorted.shape[0]),df_sorted['Close'], color = 'black', label = 'Real AAL Stock Price')
plt.plot(predicted_stock_price,color = 'red', label = 'Predicted AAL Stock Price')
plt.title('AAL Stock Price Prediction')
plt.xticks(range(0,df_sorted.shape[0],500),df_sorted['Date'].loc[::500],rotation=45)
plt.ylabel('AAL Stock Price')
plt.legend()
plt.show()