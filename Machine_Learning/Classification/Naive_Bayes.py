# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:53:22 2023

@author: gencs
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data upload and process
dataset = pd.read_csv("veriler.csv")

x = dataset.iloc[:,1:4].values
y = dataset.iloc[:,4:].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=(0))


#data scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
 
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=(0))
logr.fit(X_train, y_train.ravel())
y_pred = logr.predict(X_test)
print(y_pred)


#Confusion Matrix for rating
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


#K-Nearest Neighborhood Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")

knn.fit(X_train,y_train.ravel())
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#Support vector machine classifier
from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(X_train, y_train.ravel())

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("SVC")
print(cm)


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train.ravel())
y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("GNB")
print(cm)
