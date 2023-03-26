# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:22:13 2023

@author: gencs
"""

#Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#preprocessing
Data = pd.read_csv("Wine.csv")
X = Data.iloc[:,0:13].values
Y = Data.iloc[:,13].values
 
#train-test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=(0))

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#PCA   (stands to compare with lda)
from sklearn.decomposition import PCA
pca = PCA(n_components=(2))

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)



from sklearn.linear_model import LogisticRegression

#Before pca transform logistic regression
classifier = LogisticRegression(random_state=(0))
classifier.fit(X_train, y_train.ravel())

#After pca transform logistic regression
classifier2 = LogisticRegression(random_state=(0))
classifier2.fit(X_train2, y_train.ravel())

#Predicitions
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)


from sklearn.metrics import confusion_matrix

#actual / without pca
cm = confusion_matrix(y_test, y_pred)
print(cm)

#actual / pca
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

#without pca / pca
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=(2))

X_train_lda = lda.fit_transform(X_train, y_train.ravel())
X_test_lda = lda.transform(X_test)

#After LDA transform
classifier_lda = LogisticRegression(random_state=(0))
classifier_lda.fit(X_train_lda,y_train)

#LDA prediction
y_pred_lda = classifier_lda.predict(X_test_lda)


#After LDA / actual data
print("lda and actual")
cm4 = confusion_matrix(y_pred, y_pred_lda)
print(cm4)



