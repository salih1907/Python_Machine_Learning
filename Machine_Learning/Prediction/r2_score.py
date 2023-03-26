# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:01:54 2023

@author: gencs
"""

#1. libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
#2.  data preprocessing

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

print("Linear r2 score")
print(r2_score(Y, lin_reg.predict(X)))


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

print("Polynomial r2 score")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

#data scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_scaled = sc.fit_transform(X)
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(Y)


#SVR regression
from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_scaled, y_scaled)

plt.scatter(x_scaled, y_scaled,color="red")
plt.plot(x_scaled, svr_reg.predict(x_scaled),color="blue")
plt.show()
print("SVR r2 score")
print(r2_score(y_scaled, svr_reg.predict(x_scaled)))


#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=(0))
r_dt.fit(X,Y.ravel())

K = X - 0.4
Z = X + 0.5
plt.scatter(X, Y, color = "red")
plt.plot(x, r_dt.predict(X),color ="blue")
plt.show()

print("decision tree r2 score")
print(r2_score(Y, r_dt.predict(X)))

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators= 10, random_state = 0)
rf_reg.fit(X, Y)

print(f"random forest= {rf_reg.predict([[8]])}")

plt.scatter(X, Y, color="red")
plt.plot(X, rf_reg.predict(X), color="blue")
plt.plot(X, rf_reg.predict(Z), color="green")
plt.plot(X, r_dt.predict(K), color="yellow")


print("Random forest r2 score")
print(r2_score(Y, rf_reg.predict(X)))





