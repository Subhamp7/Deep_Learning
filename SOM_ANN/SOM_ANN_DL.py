# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:02:39 2020

@author: subham
"""


#part 1 (SOM)

#importing the required libararies
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from notification_sound import sound
from minisom import MiniSom
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


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
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#visualizing the result
from pylab import bone, pcolor , colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers= ['o', 's']
colors= ['r', 'g']
for i, x in enumerate(X): 
    w= som.winner(x)
    plot(w[0]+0.5,
        w[1]+0.5,
        markers[Y[i]],
        markeredgecolor = colors[Y[i]],
        markerfacecolor = 'None',
        markersize= 10,
        markeredgewidth=2)
show()
    
#finding the frauds
mappings = som.win_map(X)
frauds= np.concatenate((mappings[(2,1)], mappings[(3,1)]))
frauds=sc.inverse_transform(frauds)



#part 2 ANN 

customers=dataset.iloc[:,1:].values
is_fraud=np.zeros(len(dataset))
for i in range(0,len(dataset)):
    if(dataset.iloc[i,0] in frauds):
        is_fraud[i]=1

#Feature scaling 
sc=StandardScaler()
customers=sc.fit_transform(customers)

# Initialising the ANN
classifier= Sequential()
#adding first hidden layer
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu' ,input_dim=15 ))
#adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#compiling the ANN
classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])
 #fitting the model
classifier.fit(customers, is_fraud, batch_size=1, epochs=4 )
#predicting the test data
Y_pred=classifier.predict(customers)

Y_pred=np.concatenate((dataset.iloc[:,0:1].values, Y_pred), axis=1)
Y_pred=Y_pred[Y_pred[:, 1].argsort()]


#completion 
print('Complete')
sound(5)
