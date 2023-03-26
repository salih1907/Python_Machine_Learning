# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:22:02 2023

@author: gencs
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values 
Y = dataset.iloc[:,13].values

#Preprocessing
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2= LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])



#train - test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=(0))


#XGBoost classifier
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test, y_pred)

print(cm)


