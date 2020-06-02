# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:57:03 2020

@author: subham
"""
#importing the required libararies
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from notification_sound import sound
from minisom import MiniSom

#loading the dataset
dataset=pd.read_csv('Credit_Card_Applications.csv')

#checking the datset
Data_describe=dataset.describe()

#splitting the data, so that the dependent data can be used for accuracy testing 
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

#scaling 
sc= MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)

#training the SOM(Using Minisom.py)
som=MiniSom(x=10, y=10,input_len=15, sigma=1.0, learning_rate=0.5)


print('Complete')
sound(5)