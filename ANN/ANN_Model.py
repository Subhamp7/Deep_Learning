# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:43:54 2020

@author: subham
"""
#importing libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


#importing dataset
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

#encoding the categorical data
LabelEncoder_X_1=LabelEncoder()
X[:,1]=LabelEncoder_X_1.fit_transform(X[:,1])
LabelEncoder_X_2=LabelEncoder()
X[:,2]=LabelEncoder_X_2.fit_transform(X[:,2])
onehotencoder=ColumnTransformer([('one_hot', OneHotEncoder(),[1])], remainder='passthrough')
X= np.array(onehotencoder.fit_transform(X), dtype=np.float)
X=X[:,1:]

#splitting the dataset to test and train data
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=0)

#Feature scaling 
sc=StandardScaler()
X_test=sc.fit_transform(X_test)
X_train=sc.transform(X_train)

# Initialising the ANN
classifier= Sequential()
#adding first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu' ,input_dim=11 ))
#adding second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
#adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#compiling the ANN
classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])
 #fitting the model
classifier.fit(X_train, Y_train, batch_size=10, epochs=100 )
#predicting the test data
Y_pred=classifier.predict(X_test)
Y_pred=(Y_pred>0.5)
#Confusion Matrix
cm=confusion_matrix(Y_test, Y_pred)
print('confusion Matrix for the ANN model:',cm)

# Evaluating the ANN
def build_classifier():
    classifier= Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu' ,input_dim=11 ))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10, epochs=100 )
accuracies=cross_val_score(estimator= classifier, X=X_train, y=Y_train, cv= 10, n_jobs=-1) #n_jobs=-1 for using all processors
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print('Completed')