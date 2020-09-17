#import dependencies
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#reads the .csv file and creates a variable of only the close prices
df = pd.read_csv(r'C:\Users\Nathan\Desktop\PyStock\NFLX.csv')
df = df[['Close']]


#creates a varible to predict 'x' days in the future
days_in_future = 110
#Create a new column (the target or dependent variable) shifted 'x' units/days up
df['Prediction'] = df.shift(-days_in_future)


#creates a training data array
X = np.array(df.drop(['Prediction'],1))[:-days_in_future]
#creates a target data array
y = np.array(df['Prediction'])[:-days_in_future]
#splits the data into 75% training and 25% testing data sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
#Create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)
#Create the linear regression model
lr = LinearRegression().fit(x_train, y_train)
#Get the feature data, 
#AKA all the rows from the original data set except the last 'x' days
x_future = df.drop(['Prediction'], 1)[:-days_in_future]
#Get the last 'x' rows
x_future = x_future.tail(days_in_future) 
#Convert the data set into a numpy array
x_future = np.array(x_future)
#Show the model tree prediction
trpv = tree.predict(x_future)
print(trpv)
#Show the model linear regression prediction
lrpv = lr.predict(x_future)
print(lrpv)


#visualize the data
tree_predition_with_validation = trpv
lr_prediction_with_validation = lrpv
#creates a data frame with Close Price, Tree Predictions and LR predictions
valid = df[X.shape[0]:]
valid['Tree Predictions'] = tree_predition_with_validation
valid['LR Predictions'] = lr_prediction_with_validation
#sets up the chart, plots the data and displays it
plt.style.use('classic')
plt.figure(figsize=(16,8))
plt.title('NFLX (Netflix) NASDAQ Predictions with LR and Tree', fontsize = 18)
plt.xlabel('Days', fontsize= 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Tree Predictions', 'LR Predictions']])
plt.legend(['Train', 'Valid', 'Tree', 'LR'], loc='lower right')
plt.show()