# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:04:51 2023

@author: gencs
"""

#1.Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data prep.
dataset = pd.read_excel("Iris.xls")

x = dataset.iloc[:,:4].values #independent variables
y = dataset.iloc[:,4:].values #dependent variables

#Data splitting for the training and testing
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#Data scaling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Classification starts from here
# 1. Logistic Regression 

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train.ravel())#Training

y_pred = logr.predict(X_test)#prediction

#Confusion matrix for evaluation
from sklearn.metrics import confusion_matrix
print("Logistic")
cm = confusion_matrix(y_test,y_pred)
print(cm)

# 2. K-Nearest Neighborhood Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean',weights="distance")
knn.fit(X_train,y_train.ravel())

y_pred = knn.predict(X_test)

print(f"KNN, n= {knn.n_neighbors}, metric = {knn.metric}")
cm = confusion_matrix(y_test,y_pred)
print(cm)


# 3. Support Vector Machine classifier

from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train,y_train.ravel())

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(f'SVC {svc.kernel}')
print(cm)

# 4. Naive-Bayes Classifier

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train.ravel())

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

# 5. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train.ravel())
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(f'DTC crit= {dtc.criterion}')
print(cm)

# 6. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5, criterion = 'entropy')
rfc.fit(X_train,y_train.ravel())

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(f'RFC n_est= {rfc.n_estimators}')
print(cm)




