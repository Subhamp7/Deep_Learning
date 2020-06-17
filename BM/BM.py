#importing libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable

#importing the dataset
movies   =pd.read_csv('review_1m/movies.dat',  sep='::', header=None, engine='python', encoding='latin1')
users    =pd.read_csv('review_1m/users.dat',   sep='::', header=None, engine='python', encoding='latin1')
ratings  =pd.read_csv('review_1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin1')

#loading the training and the test data insted of splitting fom the above dataset
training_data   =pd.read_csv('review_100k/u1.base',  delimiter= '\t', header=None)
training_data   =np.array(training_data, dtype= 'int')
testing_data    =pd.read_csv('review_100k/u1.test',  delimiter= '\t', header=None)
testing_data    =np.array(testing_data, dtype= 'int')

#getting the total number of users and their ratings
nb_users       =int(max(max(training_data[:,0]), max(testing_data[:,0])))
nb_ratings     =int(max(max(training_data[:,1]), max(testing_data[:,1])))

#creating an array which consist of user and all his ratings
def convert(data):
    organised_data=[]
    for index in range(0, nb_users+1):
        movies_id    =data[:,1][data[:,0]==index]
        rating_id    =data[:,2][data[:,0]==index]
        rating_list  =np.zeros(nb_ratings)
        rating_list[movies_id-1]=rating_id
        organised_data.append(list(rating_list))
    return organised_data

training_data =convert(training_data)
testing_data  =convert(testing_data)

#converting the data to torch tensors
training_data =torch.FloatTensor(training_data)
testing_data  =torch.FloatTensor(testing_data)
        
#replacing the rating with -1(no rating), 0(not liked) and 1(liked).
training_data[training_data == 0] = -1
training_data[training_data == 1] =  0
training_data[training_data == 2] =  0
training_data[training_data >= 3] =  1
        
        
        
    


print('Completed')

