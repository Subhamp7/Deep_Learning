# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:13:44 2020

@author: subham
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
print('Libraries imported successfully')

#importing the training data
training_set=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=training_set.iloc[:,1:2].values

#feature scaling the training dataset
ms=MinMaxScaler()
training_set=ms.fit_transform(training_set)

#splitting the data into X and Y for training at time T and T+1
X_train=training_set[0:1257]
Y_train=training_set[1:1258]

#reshaping
X_train=np.reshape(X_train,(1257,1,1))

#applying LSTM
regressor = Sequential()
regressor.add(LSTM(units = 6, activation = 'sigmoid', input_shape =(None, 1)))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, Y_train , batch_size=32, epochs=200)
print('Model trained successfully')

#importing the test data
test_set=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test_set.iloc[:,1:2].values

#feature scaling test dataset same as train data
inputs=real_stock_price
inputs=ms.transform(inputs)

#reshaping test data
inputs=np.reshape(inputs,(20,1,1))

#predicting the T+1 data by proving the input data and inverse scaling
predicted_stock_price=regressor.predict(inputs)
predicted_stock_price=ms.inverse_transform(predicted_stock_price)

#visualizing the predicted and real data
plt.plot(real_stock_price, color='blue', label='Real_stock_price')
plt.plot(predicted_stock_price, color='red', label='Predicted_stock_price')
plt.title('Google Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

print('Completed')

