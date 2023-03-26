# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:33:32 2023

@author: gencs
"""


#1. libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. data preprocessing

#2.1. data loading
dataset = pd.read_csv('maaslar.csv')
#pd.read_csv("dataset.csv")

#dataframe slicing
x = dataset.iloc[:,1:2]
y = dataset.iloc[:,2:]
#numpy array transformations
X = x.values
Y = y.values

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)



#polynomial regression
#creating non-linear model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


#4th degree polynomial
poly_reg3 = PolynomialFeatures(degree= 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3, y)


#Visualization
plt.scatter(X, Y, color="red")
plt.plot(x,lin_reg.predict(X),color="blue")
plt.show()
plt.scatter(X, Y,color = "red")
plt.plot(X, lin_reg2.predict(x_poly),color= "blue")
plt.show()
plt.scatter(X, Y,color = "red")
plt.plot(X, lin_reg3.predict(x_poly3),color= "blue")
plt.show()

#Predictions
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))













