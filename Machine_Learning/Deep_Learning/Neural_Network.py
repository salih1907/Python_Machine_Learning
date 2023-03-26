# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:37:45 2023

@author: gencs
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("Churn_Modelling.csv")


#PREPROCESSING
X = data.iloc[:,3:13].values
Y = data.iloc[:,13:].values

#Encoding
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([('ohe',OneHotEncoder(categories='auto'),[1])],remainder='passthrough')

X = ohe.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=(0))


#Data Scaling > 0-1 interval
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#Artificial neural networks

import keras 

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, activation = "relu", input_dim = 11))#Hidden layer 1

classifier.add(Dense(6, activation = "relu"))#Hidden layer 2

classifier.add(Dense(1, activation = "sigmoid"))#Hidden layer 3

classifier.compile(optimizer="adam", loss = "binary_crossentropy", metrics=(["accuracy"]))

classifier.fit(X_train,y_train, epochs=50)
 
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)