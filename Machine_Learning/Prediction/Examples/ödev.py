# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:45:09 2023

@author: gencs
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")


from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)

#encoder: Nominal Ordinal (Kategorik) -> Numeric

from sklearn import preprocessing

outlook = veriler.iloc[:,0:1].values
le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()



#numpy dizilieri -> dataframe dönüşümleri


result1 = pd.DataFrame(data=outlook, index = range(14),columns=["o","r","s"])
result2 = pd.concat([result1,veriler.iloc[:,1:3]],axis=1)
result3 = pd.concat([veriler2.iloc[:,3:],result2],axis= 1)

play = veriler2.iloc[:,-1:].values
play = pd.DataFrame(data=play, index = range(14),columns=["play"])

#train-test
from sklearn.model_selection import train_test_split

test_data = result3.iloc[:,[1,2,3,4,5]]

x_train,x_test,y_train,y_test = train_test_split(test_data,result3.iloc[:,-1:],test_size=0.33,random_state=(0))

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


#BACKWARD ELIMINATION
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values= result3.iloc[:,:-1], axis = 1)

X_list = test_data.values
X_list = np.array(X_list,dtype=(float))
model = sm.OLS(result3.iloc[:,-1:], X_list).fit()
print(model.summary())



1

 