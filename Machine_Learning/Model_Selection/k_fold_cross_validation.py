# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:48:22 2023

@author: gencs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#preprocessing
dataset = pd.read_csv("Social_Network_Ads.csv") 

X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

#train - test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=(0))

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf",random_state=(0))
classifier.fit(X_train, y_train.ravel())

#Prediction
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#K-fold Cross Validation

from sklearn.model_selection import cross_val_score
cross_vs = cross_val_score(estimator= classifier, X = X_train, y= y_train, cv= 4)
print(cross_vs.mean())





