# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:24:04 2020

@author: subham
"""

# Plot ad hoc mnist instances
from keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  pop_up import index_name

# load (downloaded if needed) the MNIST dataset
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#OR
# load the downloaded dataset
dataset_train=pd.read_csv('mnist_train.csv')
dataset_test=pd.read_csv('mnist_test.csv')

#displaying the digit count for both train and test data
number_values_train=dataset_train.iloc[:,:1].groupby('label')['label'].count()
number_values_test=dataset_test.iloc[:,:1].groupby('label')['label'].count()
print(('The amount of digits in training set is: \n {} \n The amount of digits in test set is: \n {} ').
      format(number_values_train,number_values_test))

#plotting the digit count for both train and test data
fig, ax = plt.subplots(figsize=(8,5))
p1 = ax.bar(number_values_train.index,number_values_train.values)
p2 = ax.bar(number_values_test.index,number_values_test.values)
l = ax.legend([p1,p2],['Train data', 'Test data'])
plt.xlabel("Digits available")
plt.ylabel("Frequency of the Digit")
plt.title("Digits and their count") 

#splitting the training data
dataset_train_X=np.asarray(dataset_train.iloc[:,1:]).reshape([len(dataset_train), 28, 28, 1])
dataset_train_Y=np.asarray(dataset_train.iloc[:,:1]).reshape([len(dataset_train), 1])

#splitting the test data
dataset_test_X=np.asarray(dataset_test.iloc[:,1:]).reshape([len(dataset_test), 28, 28, 1])
dataset_test_Y=np.asarray(dataset_test.iloc[:,:1]).reshape([len(dataset_test), 1])

#converting pixel value in the range 0 to 1
dataset_train_X=dataset_train_X/255
dataset_test_X =dataset_test_X/255

#visualizing some of the digits
index=index_name()
plt.imshow(dataset_train_X[index].reshape([28,28]),cmap="Blues")
plt.title(('The number is:',str(dataset_train_Y[index])), y=-0.15,color="green")







