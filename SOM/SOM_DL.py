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
frauds= np.concatenate((mappings[(8,6)], mappings[(6,8)]))
frauds=sc.inverse_transform(frauds)

#completion 
print('Complete')
sound(5)